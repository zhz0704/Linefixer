import sys
argv = sys.argv[1:]

print(f"Running inference script with args:\n{'\n'.join(f"{argv[i]}, {argv[i+1]}" for i in range(0, len(argv), 2))}")


import os
from pathlib import Path
import argparse

import torch
from PIL import Image
from tqdm import tqdm

from linefixer.models.unet import UNet
from linefixer.utils.patch_infer import fix_line_art


def main():
    parser = argparse.ArgumentParser(description='Run inference over test/broken and write to test/fixed')
    parser.add_argument('--test_root', type=str, default='test', help="Root test directory containing 'original' and 'fixed'")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth', help='Model checkpoint path')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--overlap', type=int, default=0)
    parser.add_argument('--display_every', type=int, default=200, help='Display every Nth image (0 to disable)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Model
    model = UNet(n_channels=1, n_classes=1, bilinear=True).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', -1) + 1}")

    broken_root = Path(args.test_root) / 'broken'
    fixed_root = Path(args.test_root) / 'fixed'

    for category in sorted([d for d in broken_root.iterdir() if d.is_dir()]):
        print(f"Processing {category.name}")
        input_dir = category
        output_dir = fixed_root / category.name
        output_dir.mkdir(parents=True, exist_ok=True)

        png_files = sorted([f for f in input_dir.iterdir() if f.suffix.lower() == '.png'])
        for i, file in tqdm(list(enumerate(png_files)), total=len(png_files)):
            output_image = fix_line_art(
                model,
                str(file),
                device=device,
                patch_size=args.image_size,
                threshold=args.threshold,
                overlap=args.overlap,
                display=(args.display_every > 0 and (i % args.display_every == 0)),
            )
            output_image.save(str(output_dir / file.name))


if __name__ == '__main__':
    main()
