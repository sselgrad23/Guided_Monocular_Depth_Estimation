#!/usr/bin/env python
import argparse, os, math, csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL
from PIL import Image

# --- your project modules ---
from load_metric import *        # uses your exact sparse-depth normalization/stacking
from vis import visualize_depth                 # your renderer that takes a .npy path and writes a PNG

# Optional: SSIM (won't crash if not installed)
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SSIM = True
except Exception:
    HAS_SSIM = False


def psnr_from_mse(mse, max_i=2.0):  # inputs in [-1,1] → dynamic range = 2
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(max_i) - 10.0 * math.log10(mse)


def make_side_by_side(left_png: str, right_png: str, out_png: str, pad_color=(0, 0, 0)):
    """Horizontally concatenate two PNGs, letterboxing the shorter one if heights differ."""
    L = Image.open(left_png).convert("RGB")
    R = Image.open(right_png).convert("RGB")
    h = max(L.height, R.height)

    def letterbox(im):
        if im.height == h:
            return im
        canvas = Image.new("RGB", (im.width, h), pad_color)
        top = (h - im.height) // 2
        canvas.paste(im, (0, top))
        return canvas

    Lb, Rb = letterbox(L), letterbox(R)
    combo = Image.new("RGB", (Lb.width + Rb.width, h), pad_color)
    combo.paste(Lb, (0, 0))
    combo.paste(Rb, (Lb.width, 0))
    combo.save(out_png)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser("VAE test on sparse depth only (with visualization)")
    ap.add_argument("--pretrained_model_name_or_path", required=True,
                    help="Base model path or HF id; VAE is loaded from its 'vae' subfolder.")
    ap.add_argument("--dataset", choices=["hypersim", "vkitti"], required=True)

    # dataset roots
    ap.add_argument("--hypersim_data_dir", type=str, default=None)
    ap.add_argument("--hypersim_csv_filepath", type=str, default=None)
    ap.add_argument("--vkitti_data_dir", type=str, default=None)

    # runtime
    ap.add_argument("--limit", type=int, default=128, help="Max samples to evaluate (None = full set).")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda")

    # viz/output
    ap.add_argument("--outdir", type=str, default="vae_sparse_out")
    ap.add_argument("--cmap", type=str, default="Spectral")
    ap.add_argument("--point_size", type=float, default=2.5)
    ap.add_argument("--side-by-side", action="store_true",
                    help="Also write *_cmp.png with input vs reconstruction.")
    ap.add_argument("--keep-individual-pngs", action="store_true",
                    help="Keep separate *_in.png and *_rec.png even when --side-by-side is used.")
    ap.add_argument("--keep-npy", action="store_true",
                    help="Keep intermediate .npy files (by default they are deleted to save space).")
    ap.add_argument("--sparse_depth_name", type=str, default=None,
                    help="Subdirectory name for sparse depth maps (default: None).")

    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    outdir = Path(args.outdir)
    (outdir / "npy").mkdir(parents=True, exist_ok=True)
    (outdir / "viz").mkdir(parents=True, exist_ok=True)

    # ---- dataset (uses your loader & normalization for sparse depth) ----
    if args.dataset == "hypersim":
        if not (args.hypersim_data_dir and args.hypersim_csv_filepath):
            ap.error("--hypersim_data_dir and --hypersim_csv_filepath are required for --dataset hypersim")
        ds = Hypersim(root_dir=args.hypersim_data_dir,
                      csv_filepath=args.hypersim_csv_filepath,
                      transform=True)
        near, far = ds.near_plane, ds.far_plane
    else:
        if not args.vkitti_data_dir:
            ap.error("--vkitti_data_dir is required for --dataset vkitti")
        ds = VirtualKITTI2(root_dir=args.vkitti_data_dir, transform=True, sparse_depth_name=args.sparse_depth_name)
        near, far = ds.near_plane, ds.far_plane

    # Optionally cap the dataset to the first N samples (deterministic)
    if args.limit is not None:
        class _Subset(torch.utils.data.Dataset):
            def __init__(self, base, n): self.base, self.n = base, min(n, len(base))
            def __len__(self): return self.n
            def __getitem__(self, i): return self.base[i]
        ds = _Subset(ds, args.limit)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # ---- VAE ----
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,
                                        subfolder="vae").to(device).eval()
    scaling = getattr(vae.config, "scaling_factor", 0.18215)

    # IMPORTANT: must match your loader's epsilon used in depth normalization
    EPS = 0.02

    # Accumulators
    mse_all, mae_all, psnr_all, ssim_all, kl_all = [], [], [], [], []
    rows = []
    idx_global = 0

    def invert_to_metric(d_norm_3ch, valid_mask):
        """
        Invert your normalization back to meters.
        - d_norm_3ch: Bx3xHxW in [-1,1]
        - valid_mask: BxHxW boolean (True where sparse point exists)
        Returns metric depth BxHxW (meters), invalid set to 0.
        """
        d_norm = d_norm_3ch[:, 0, :, :]
        d01 = (d_norm + 1.0) * 0.5                          # [-1,1] -> [0,1]
        raw01 = torch.zeros_like(d01)
        # forward was: x01_valid = raw01*(1-EPS) + EPS  ⇒ raw01 = (x01_valid - EPS)/(1-EPS)
        raw01[valid_mask] = torch.clamp((d01[valid_mask] - EPS) / (1.0 - EPS), 0.0, 1.0)
        metric = near + raw01 * (far - near)
        metric = metric * valid_mask                        # zero invalid
        return metric

    def encode_decode(x_sparse):
        """Deterministic VAE recon using posterior mean."""
        enc = vae.encode(x_sparse)
        mu = enc.latent_dist.mean
        if hasattr(enc.latent_dist, "logvar"):
            logvar = enc.latent_dist.logvar
            kl = 0.5 * (torch.exp(logvar) + mu**2 - 1.0 - logvar).mean(dim=(1, 2, 3))
        else:
            kl = torch.zeros(x_sparse.size(0), device=x_sparse.device)
        lat = mu * scaling
        recon = vae.decode(lat / scaling).sample             # in [-1,1]
        return recon, kl

    def masked_metrics(x, x_rec, vmask):
        """Compute MSE/MAE/PSNR on valid sparse pixels only (normalized domain)."""
        diff = x_rec - x
        m3 = vmask[:, None, :, :].expand_as(x)
        pix = m3.sum(dim=(1, 2, 3)).clamp_min(1)
        mse = ((diff ** 2) * m3).sum(dim=(1, 2, 3)) / pix
        mae = (diff.abs() * m3).sum(dim=(1, 2, 3)) / pix
        ps = torch.tensor([psnr_from_mse(float(m)) for m in mse], device=x.device)
        ssim_vals = []
        if HAS_SSIM:
            # Convert to [0,1] HxWxC
            x01 = (x.permute(0, 2, 3, 1) + 1.0) * 0.5
            r01 = (x_rec.permute(0, 2, 3, 1) + 1.0) * 0.5
            for i in range(x01.size(0)):
                s = ssim(x01[i].cpu().numpy(), r01[i].cpu().numpy(),
                         channel_axis=-1, data_range=1.0)
                ssim_vals.append(float(s))
        return mse.cpu().tolist(), mae.cpu().tolist(), ps.cpu().tolist(), ssim_vals

    for bidx, batch in enumerate(dl):
        # Your loader exposes sparse depth already normalized to [-1,1] and stacked to 3 channels
        x_sparse = batch["depth"].to(device, dtype=torch.float32)  # Bx3xHxW #Formerly sparse_depth
        B = x_sparse.size(0)

        # Valid sparse mask (invalid sparse → exactly -1 after normalization). Use tolerance for safety.
        vmask = (x_sparse[:, 0, :, :] > -0.9999)

        # VAE recon
        recon, kl_vec = encode_decode(x_sparse)

        # Metrics on valid sparse pixels only
        mse, mae, psnr, ssim_vals = masked_metrics(x_sparse, recon, vmask)
        mse_all += mse
        mae_all += mae
        psnr_all += psnr
        kl_all += kl_vec.cpu().tolist()
        if HAS_SSIM:
            ssim_all += ssim_vals

        # Invert to metric for visualization (input & recon)
        metric_in = invert_to_metric(x_sparse, vmask)
        metric_rec = invert_to_metric(recon, vmask)

        # Save + visualize + cleanup
        for i in range(B):
            i_global = idx_global + i

            # 1) Save metric depths as .npy for your visualize_depth()
            npy_in = outdir / "npy" / f"sparse_in_{i_global:06d}.npy"
            npy_rec = outdir / "npy" / f"sparse_rec_{i_global:06d}.npy"
            np.save(npy_in, metric_in[i].cpu().numpy())
            np.save(npy_rec, metric_rec[i].cpu().numpy())

            # 2) Render PNGs using your painter
            png_in = outdir / "viz" / f"sparse_in_{i_global:06d}.png"
            png_rec = outdir / "viz" / f"sparse_rec_{i_global:06d}.png"
            visualize_depth(str(npy_in), str(png_in), cmap=args.cmap, point_size=args.point_size)
            visualize_depth(str(npy_rec), str(png_rec), cmap=args.cmap, point_size=args.point_size)

            # 3) Side-by-side comparison (optional)
            if args.side_by_side:
                cmp_png = outdir / "viz" / f"depth_cmp_{i_global:06d}.png"
                make_side_by_side(str(png_in), str(png_rec), str(cmp_png))
                if not args.keep_individual_pngs:
                    try:
                        os.remove(png_in)
                        os.remove(png_rec)
                    except OSError:
                        pass

            # 4) Delete .npy to save space (unless user keeps them)
            if not args.keep_npy:
                try:
                    os.remove(npy_in)
                    os.remove(npy_rec)
                except OSError:
                    pass

            # Per-image CSV row
            rows.append({
                "idx": i_global,
                "depth_mse": mse[i],
                "depth_mae": mae[i],
                "depth_psnr": psnr[i],
                "depth_ssim": (ssim_vals[i] if HAS_SSIM else None),
                "kl": float(kl_vec[i].item()),
            })

        idx_global += B

    # --- aggregate printout ---
    def agg(x):
        x = np.array(x, dtype=np.float64)
        return dict(mean=float(np.mean(x)),
                    median=float(np.median(x)),
                    p90=float(np.percentile(x, 90)),
                    p95=float(np.percentile(x, 95)))

    print("\n=== VAE RECON (DEPTH, normalized, masked by valid points) ===")
    print("MSE  :", agg(mse_all))
    print("MAE  :", agg(mae_all))
    print("PSNR :", agg(psnr_all))
    if HAS_SSIM:
        print("SSIM :", agg(ssim_all))
    print("KL   :", agg(kl_all))

    # --- write per-image CSV ---
    if rows:
        csv_path = outdir / "per_image_metrics_depth.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nPer-image metrics → {csv_path}")


if __name__ == "__main__":
    main()
