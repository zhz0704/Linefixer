from typing import Dict
import matplotlib.pyplot as plt
import torch


def visualize_batch(epoch, model, loader, device, num_samples=4, results_dir=None):
    """Visualize model predictions from a batch.

    Args:
        model: PyTorch model
        loader: DataLoader to draw a batch from
        device: torch.device
        num_samples: number of rows to visualize
    """
    model.eval()

    with torch.no_grad():
        batch = next(iter(loader))
        broken, fixed = batch[0].to(device), batch[1].to(device)
        predictions = model(broken)

        fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 3))
        if num_samples == 1:
            axes = [axes]

        for i in range(min(num_samples, broken.size(0))):
            # Broken input
            axes[i][0].imshow(broken[i, 0].cpu(), cmap='gray')
            axes[i][0].set_title('Broken Input')
            axes[i][0].axis('off')

            # Model prediction
            axes[i][1].imshow(predictions[i, 0].cpu(), cmap='gray')
            axes[i][1].set_title('Model Output')
            axes[i][1].axis('off')

            # Ground truth
            axes[i][2].imshow(fixed[i, 0].cpu(), cmap='gray')
            axes[i][2].set_title('Ground Truth')
            axes[i][2].axis('off')

        plt.tight_layout()
        if results_dir:
            plt.savefig(f"{results_dir}/visualization_epoch_{epoch}.png")
        else:
            plt.show()


def plot_training_history(epoch, history: Dict[str, list], results_dir=None):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    ax1.plot(history.get('train_loss', []), label='Train Loss')
    ax1.plot(history.get('val_loss', []), label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Learning rate
    ax2.plot(history.get('lr', []))
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.grid(True)

    plt.tight_layout()
    if results_dir:
        plt.savefig(f"{results_dir}/training_history_epoch_{epoch}.png")
    else:
        plt.show()


def compare(test_dir='test', num_samples=3, random_seed=42):
    """
    Compare original and fixed images side by side from test directories

    Args:
        test_dir: Root test directory containing 'original' and 'fixed' subdirs
        num_samples: Number of random samples to show from each category
        random_seed: Random seed for reproducibility
    """
    import os
    from PIL import Image
    import random

    random.seed(random_seed)

    original_root = os.path.join(test_dir, 'original')
    fixed_root = os.path.join(test_dir, 'fixed')

    if not (os.path.isdir(original_root) and os.path.isdir(fixed_root)):
        raise FileNotFoundError(f"Expected '{original_root}' and '{fixed_root}' directories to exist.")

    # Get all subdirectories in test/broken
    categories = sorted([
        d for d in os.listdir(original_root)
        if os.path.isdir(os.path.join(original_root, d))
    ])

    if len(categories) == 0:
        raise FileNotFoundError(f"No category subdirectories found in {original_root}")

    # Create a figure: one row per category, two columns per sample (orig/fixed)
    import math
    cols = 2 * max(1, num_samples)
    fig, axes = plt.subplots(len(categories), cols, figsize=(4 * num_samples, 3 * len(categories)))
    if len(categories) == 1:
        axes = [axes]

    for cat_idx, category in enumerate(categories):
        # Get list of images in this category
        orig_dir = os.path.join(original_root, category)
        fixed_dir = os.path.join(fixed_root, category)
        images = sorted([f for f in os.listdir(orig_dir) if f.lower().endswith('.png')])

        if len(images) == 0:
            continue

        selected_images = random.sample(images, min(num_samples, len(images)))

        for img_idx, img_name in enumerate(selected_images):
            # Load original and fixed image
            orig_path = os.path.join(orig_dir, img_name)
            fixed_path = os.path.join(fixed_dir, img_name)
            if not os.path.exists(fixed_path):
                continue

            orig_img = Image.open(orig_path).convert('L')
            fixed_img = Image.open(fixed_path).convert('L')

            # Plot original
            ax = axes[cat_idx][img_idx * 2]
            ax.imshow(orig_img, cmap='gray')
            ax.axis('off')
            title = f'{category}\nOriginal' if img_idx == 0 else 'Original'
            ax.set_title(title, size=10)

            # Plot fixed
            ax = axes[cat_idx][img_idx * 2 + 1]
            ax.imshow(fixed_img, cmap='gray')
            ax.axis('off')
            ax.set_title('Fixed', size=10)

    plt.tight_layout()
    plt.show()
