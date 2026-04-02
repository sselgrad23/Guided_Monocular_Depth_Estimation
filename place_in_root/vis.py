import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import Normalize
from PIL import Image
from tqdm import tqdm

def visualize_depth(input_path, output_path, cmap='Spectral', point_size=2.5):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if input_path.endswith('.npy'):
        depth = np.load(input_path)
    elif input_path.endswith('.png'):
        depth = np.array(Image.open(input_path)).astype(np.float32)
    else:
        return
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    colored = np.zeros((*depth.shape, 4))  # RGBA
    # Normalize non-zero values and mask zeros
    nonzero_mask = depth > 0
    if not np.any(nonzero_mask):
        # Empty image
        plt.imsave(output_path, np.zeros_like(depth), cmap='gray')
        return

    norm = Normalize(vmin=np.percentile(depth[nonzero_mask], 2), vmax=np.percentile(depth[nonzero_mask], 98))


    cmap_func = plt.get_cmap(cmap)
    colored[nonzero_mask, :] = cmap_func(norm(depth[nonzero_mask]))

    if np.count_nonzero(nonzero_mask) < 0.05 * depth.size:
        # Sparse mode
        height, width = depth.shape
        dpi = 100
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.imshow(np.zeros_like(depth), cmap='gray')
        y, x = np.nonzero(nonzero_mask)
        colors = cmap_func(norm(depth[nonzero_mask]))
        ax.scatter(x, y, c=colors, s=point_size**2, marker='o')
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)
    else:
        img_uint8 = (colored[:, :, :3] * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8)
        img_pil.save(output_path)

def collect_and_process(input_dir, output_dir, cmap='Spectral', point_size=2.5):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    all_paths = list(input_dir.rglob('*'))
    for path in tqdm(all_paths, desc="Visualizing", unit="file"):
        if path.suffix.lower() not in ['.png', '.npy']:
            continue
        rel_path = path.relative_to(input_dir).with_suffix('.png')
        out_path = output_dir / rel_path
        visualize_depth(str(path), str(out_path), cmap=cmap, point_size=point_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize depth or sparse maps with colormap.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_dir", type=str, help="Directory of .png/.npy files")
    group.add_argument("--input_file", type=str, help="Path to a single .npy or .png file")

    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save visualizations")
    parser.add_argument("--cmap", type=str, default="Spectral", help="Matplotlib colormap")
    parser.add_argument("--point_size", type=float, default=2.5, help="Size of circular points for sparse maps")

    args = parser.parse_args()

    if args.input_file:
        out_file = Path(args.output_dir) / (Path(args.input_file).stem + ".png")
        visualize_depth(args.input_file, str(out_file), cmap=args.cmap, point_size=args.point_size)
    else:
        collect_and_process(args.input_dir, args.output_dir, cmap=args.cmap, point_size=args.point_size)
