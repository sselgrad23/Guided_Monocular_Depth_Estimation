import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from tabulate import tabulate
import torch
from metric import (
    MetricTracker,
    abs_relative_difference,
    squared_relative_difference,
    rmse_linear,
    rmse_log,
    log10,
    delta1_acc,
    delta2_acc,
    delta3_acc,
    i_rmse,
    silog_rmse,
)
import pdb
import csv

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate predicted depth maps against ground truth.")
    parser.add_argument("--file_list", type=str, required=True)
    parser.add_argument("--pred_root", type=str, required=True)
    parser.add_argument("--gt_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--depth_min", type=float, default=0.6, help="Minimum depth in meters")
    parser.add_argument("--depth_max", type=float, default=350.0, help="Maximum depth in meters")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="diode",
        choices=["diode", "kitti", "eth_3d", "nyu_depth_v2", "scannet", "vkitti"],
        help="Name of the dataset (affects depth scale handling)",
    )
    parser.add_argument("--identity_test", action="store_true",
                        help="Use GT as both pred and GT for sanity checking")
    return parser.parse_args()


def load_pred(path):
    if path.endswith(".npy"):
        pred = np.load(path).astype(np.float32)
    else:
        raise ValueError(f"Unsupported prediction file format: {path}")
    return pred


def load_gt(path, dataset_name):
    if dataset_name in ["nyu_depth_v2", "scannet"]:
        gt_raw = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        gt = gt_raw / 1000.0  # mm → meters
    elif dataset_name == "diode":
        gt = np.load(path).astype(np.float32)
    elif dataset_name == "kitti":
        gt_raw = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) #
        gt = gt_raw / 256.0
    elif dataset_name == "vkitti":
        gt_raw = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        gt = gt_raw / 100.0
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    if gt.ndim == 3 and gt.shape[-1] == 1:
            gt = gt.squeeze(-1)
    return gt

def align_depth_least_square(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask_arr: np.ndarray,
    return_scale_shift=True,
    max_resolution=None,
):
    ori_shape = pred_arr.shape  # input shape

    gt = gt_arr.squeeze()  # [H, W]
    pred = pred_arr.squeeze()
    valid_mask = valid_mask_arr.squeeze()

    # Downsample
    if max_resolution is not None:
        scale_factor = np.min(max_resolution / np.array(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
            gt = downscaler(torch.as_tensor(gt).unsqueeze(0)).numpy()
            pred = downscaler(torch.as_tensor(pred).unsqueeze(0)).numpy()
            valid_mask = (
                downscaler(torch.as_tensor(valid_mask).unsqueeze(0).float())
                .bool()
                .numpy()
            )

    assert (
        gt.shape == pred.shape == valid_mask.shape
    ), f"{gt.shape}, {pred.shape}, {valid_mask.shape}"

    gt_masked = gt[valid_mask].reshape((-1, 1))
    pred_masked = pred[valid_mask].reshape((-1, 1))

    # numpy solver
    _ones = np.ones_like(pred_masked)
    A = np.concatenate([pred_masked, _ones], axis=-1)
    X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
    scale, shift = X

    aligned_pred = pred_arr * scale + shift

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)

    if return_scale_shift:
        return aligned_pred, scale, shift
    else:
        return aligned_pred

def base_get_valid_mask(depth_min, depth_max, depth: torch.Tensor):
    valid_mask = torch.logical_and(
        (depth > depth_min), (depth < depth_max)
    ).bool()
    return valid_mask

def kitti_benchmark_crop(input_img):
    """
    Crop images to KITTI benchmark size
    Args:
        `input_img` (torch.Tensor): Input image to be cropped.

    Returns:
        torch.Tensor:Cropped image.
    """
    KB_CROP_HEIGHT = 352
    KB_CROP_WIDTH = 1216

    height, width = input_img.shape[-2:]
    top_margin = int(height - KB_CROP_HEIGHT)
    left_margin = int((width - KB_CROP_WIDTH) / 2)
    if 2 == len(input_img.shape):
        out = input_img[
            top_margin : top_margin + KB_CROP_HEIGHT,
            left_margin : left_margin + KB_CROP_WIDTH,
        ]
    elif 3 == len(input_img.shape):
        out = input_img[
            :,
            top_margin : top_margin + KB_CROP_HEIGHT,
            left_margin : left_margin + KB_CROP_WIDTH,
        ]
    return out

def kitti_get_valid_mask(depth_min, depth_max, valid_mask_crop, depth: torch.Tensor):
    # reference: https://github.com/cleinc/bts/blob/master/pytorch/bts_eval.py
    valid_mask = base_get_valid_mask(depth_min, depth_max, depth)  # [1, H, W]

    if valid_mask_crop is not None:
        eval_mask = torch.zeros_like(valid_mask.squeeze()).bool()
        gt_height, gt_width = eval_mask.shape

        if "garg" == valid_mask_crop:
            eval_mask[
                int(0.40810811 * gt_height) : int(0.99189189 * gt_height),
                int(0.03594771 * gt_width) : int(0.96405229 * gt_width),
            ] = 1
        elif "eigen" == valid_mask_crop:
            eval_mask[
                int(0.3324324 * gt_height) : int(0.91351351 * gt_height),
                int(0.0359477 * gt_width) : int(0.96405229 * gt_width),
            ] = 1

        eval_mask = eval_mask.reshape(valid_mask.shape)
        valid_mask = torch.logical_and(valid_mask, eval_mask)
    return valid_mask

def nyuv2_get_valid_mask(depth_min, depth_max, eigen_valid_mask, depth: torch.Tensor):
    valid_mask = base_get_valid_mask(depth_min, depth_max, depth)

    # Eigen crop for evaluation
    if eigen_valid_mask:
        eval_mask = torch.zeros_like(valid_mask.squeeze()).bool()
        eval_mask[45:471, 41:601] = 1
        eval_mask.reshape(valid_mask.shape)
        valid_mask = torch.logical_and(valid_mask, eval_mask)

    return valid_mask

def main(args):
    # unpack everything from args
    file_list    = args.file_list
    pred_root    = args.pred_root
    gt_root      = args.gt_root
    output_dir   = args.output_dir
    dataset_name = args.dataset_name
    depth_min    = args.depth_min
    depth_max    = args.depth_max

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_funcs = [
        abs_relative_difference,
        squared_relative_difference,
        rmse_linear,
        rmse_log,
        log10,
        delta1_acc,
        delta2_acc,
        delta3_acc,
        i_rmse,
        silog_rmse,
    ]
    metric_tracker = MetricTracker(*[f.__name__ for f in metric_funcs])
    metric_tracker.reset()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        per_sample_path = os.path.join(output_dir, "per_sample_metrics.csv")
        summary_path = os.path.join(output_dir, "eval_metrics.txt")
        f_out = open(per_sample_path, "w")
        # header with all metric names
        f_out.write("filename," + ",".join(m.__name__ for m in metric_funcs) + "\n")
    else:
        f_out = None

    with open(file_list, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Evaluating predictions", unit="img"):
        rgb_rel, gt_rel, _, mask_rel = line.strip().split()
        base = os.path.splitext(os.path.basename(rgb_rel))[0]
        subdir = os.path.relpath(os.path.dirname(rgb_rel), start=gt_root)

        ext = "npy"
        pred_path = os.path.join(pred_root, subdir, f"{base}.{ext}")
        gt_path = os.path.join(gt_root, gt_rel)

        gt = load_gt(gt_path, dataset_name)
        if args.identity_test:
            pred = gt.copy()
        else:
            pred = load_pred(pred_path)


        if dataset_name == "kitti":
            pred = torch.from_numpy(pred).to(device)
            gt = torch.from_numpy(gt).to(device)

            pred = kitti_benchmark_crop(pred)
            gt = kitti_benchmark_crop(gt)
            mask = kitti_get_valid_mask(depth_min, depth_max, "eigen", gt)

            pred = pred.cpu().numpy()
            gt = gt.cpu().numpy()
            mask = mask.cpu().numpy()
        elif dataset_name == "nyu_depth_v2":
            mask = nyuv2_get_valid_mask(depth_min, depth_max, True, torch.from_numpy(gt))
            mask = mask.cpu().numpy()
        elif dataset_name == "vkitti":
            pred = torch.from_numpy(pred).to(device)
            gt = torch.from_numpy(gt).to(device)

            pred = kitti_benchmark_crop(pred)
            gt = kitti_benchmark_crop(gt)
            mask = base_get_valid_mask(depth_min, depth_max, gt).cpu().numpy()

            pred = pred.cpu().numpy()
            gt = gt.cpu().numpy()
        else:
            mask = base_get_valid_mask(depth_min, depth_max, torch.from_numpy(gt)).numpy()


        pred, scale, shift = align_depth_least_square(
                gt_arr=gt,
                pred_arr=pred,
                valid_mask_arr=mask,
                return_scale_shift=True,
                max_resolution=None,
            )

        # Clip prediction to valid depth range
        pred = np.clip(pred, a_min=depth_min, a_max=depth_max)

        # clip to d > 0 for evaluation
        pred = np.clip(pred, a_min=1e-6, a_max=None)

        depth_pred_ts = torch.from_numpy(pred).to(device)
        depth_gt_ts   = torch.from_numpy(gt).to(device)
        valid_mask_ts = torch.from_numpy(mask).to(device)

        # compute all metrics and update tracker
        sample_vals = []
        for fn in metric_funcs:
            val = fn(depth_pred_ts, depth_gt_ts, valid_mask_ts).item()
            sample_vals.append(f"{repr(val)}") # repr() in cluster version val:.6f
            metric_tracker.update(fn.__name__, val)

        # write per-sample results
        f_out.write(f"{base}," + ",".join(sample_vals) + "\n")

    if f_out:
        f_out.close()

    # Print evaluation summary from MetricTracker
    # Counts are the number of samples processed
    counts = metric_tracker._data["counts"]
    num_samples = int(counts.iloc[0])
    print(f"\nEvaluated {num_samples} images")

    stats = metric_tracker.result()
    for name, mean_val in stats.items():
        print(f"Mean {name}: {repr(mean_val)}") # repr() in cluster version mean_val:.4f

    if output_dir:
        with open(summary_path, "w") as f:
            # NEW: use MetricTracker’s aggregated means
            stats = metric_tracker.result()
            table = tabulate(
                [(k, str(stats[k])) for k in stats],
                headers=["Metric","Mean"],
                tablefmt="github",
                disable_numparse=True                    
            )
            f.write("Evaluation Summary:\n\n")
            f.write(table)
            # metrics + values file
            csv_path = os.path.join(output_dir, "eval_metrics.csv")
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Metric", "Mean"])
                for name, mean_val in stats.items():
                    writer.writerow([name, repr(mean_val)])

            # numbers-only file
            csv_numbers_path = os.path.join(output_dir, "num_only_eval_metrics.csv")
            with open(csv_numbers_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([repr(v) for v in stats.values()])


if __name__ == "__main__":
    args = parse_args()
    main(args)