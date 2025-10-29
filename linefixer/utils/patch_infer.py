from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt


def create_weight_mask(patch_size: int, overlap: int):
    """Create a weight mask for smooth blending of overlapping patches."""
    if overlap == 0:
        return np.ones((patch_size, patch_size))

    ramp = np.linspace(0, 1, overlap)
    mask = np.ones((patch_size, patch_size))

    mask[:overlap, :] *= ramp.reshape(-1, 1)             # Top
    mask[-overlap:, :] *= ramp[::-1].reshape(-1, 1)      # Bottom
    mask[:, :overlap] *= ramp.reshape(1, -1)             # Left
    mask[:, -overlap:] *= ramp[::-1].reshape(1, -1)      # Right

    return mask


def create_gaussian_weight_mask(patch_size: int):
    """Create a Gaussian weight mask for very smooth blending."""
    center = patch_size // 2
    sigma = patch_size / 4

    y, x = np.ogrid[:patch_size, :patch_size]
    mask = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
    return mask


def fix_line_art(model, image_path: str, device: torch.device, patch_size: int,
                 threshold: float = 0.5, overlap: int = 32, display: bool = True) -> Image.Image:
    """
    Fix a single line art image using patch-based processing.

    Args:
        model: Trained model
        image_path: Path to the broken line art image (PNG)
        device: torch.device
        patch_size: Size of square patches fed to the model
        threshold: Threshold for binarization
        overlap: Overlap between patches (pixels)
        display: If True, display original, raw output, and thresholded output
    Returns:
        Thresholded PIL Image
    """
    model.eval()

    img = Image.open(image_path).convert('L')
    img_array = np.array(img) / 255.0
    original_h, original_w = img_array.shape

    # If image is smaller than or equal to patch size, process directly with optional padding
    if original_h <= patch_size and original_w <= patch_size:
        pad_h = max(0, patch_size - original_h)
        pad_w = max(0, patch_size - original_w)
        padded = np.pad(img_array, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=1)

        img_tensor = torch.FloatTensor(padded).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor).squeeze().cpu().numpy()
        output = output[:original_h, :original_w]
    else:
        stride = patch_size - overlap
        n_h = (original_h - patch_size) // stride + 1
        if (original_h - patch_size) % stride != 0:
            n_h += 1
        n_w = (original_w - patch_size) // stride + 1
        if (original_w - patch_size) % stride != 0:
            n_w += 1

        pad_h = (n_h - 1) * stride + patch_size - original_h
        pad_w = (n_w - 1) * stride + patch_size - original_w
        if pad_h > 0 or pad_w > 0:
            img_array = np.pad(img_array, ((0, pad_h), (0, pad_w)), mode='edge')

        output = np.zeros_like(img_array)
        weights = np.zeros_like(img_array)
        weight_mask = create_weight_mask(patch_size, overlap)

        patch_count = 0
        for i in range(n_h):
            for j in range(n_w):
                y_start = i * stride
                x_start = j * stride
                y_end = y_start + patch_size
                x_end = x_start + patch_size

                patch = img_array[y_start:y_end, x_start:x_end]
                patch_tensor = torch.FloatTensor(patch).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    patch_output = model(patch_tensor).squeeze().cpu().numpy()

                output[y_start:y_end, x_start:x_end] += patch_output * weight_mask
                weights[y_start:y_end, x_start:x_end] += weight_mask

                patch_count += 1

        output = np.divide(output, weights, where=weights > 0)
        output = output[:original_h, :original_w]

    output_binary = (output > threshold).astype(np.uint8) * 255
    output_img = Image.fromarray(output_binary)

    if display:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title(f'Original (Broken) - {original_w}x{original_h}')
        axes[0].axis('off')

        axes[1].imshow(output, cmap='gray')
        axes[1].set_title('Model Output (Raw)')
        axes[1].axis('off')

        axes[2].imshow(output_img, cmap='gray')
        axes[2].set_title(f'Fixed (Threshold={threshold})')
        axes[2].axis('off')
        plt.tight_layout()
        plt.show()

    return output_img
