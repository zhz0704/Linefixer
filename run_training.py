import sys
argv = sys.argv[1:]

print(f"Running training script with args:\n{'\n'.join(f"{argv[i]}, {argv[i+1]}" for i in range(0, len(argv), 2))}")

import os
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from linefixer.models.unet import UNet
from linefixer.data.dataset import LineArtDataset
from linefixer.losses.loss import CombinedLoss
from linefixer.utils.visualize import visualize_batch, plot_training_history


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(loader, desc='Training', leave=False)
    for broken, fixed in progress_bar:
        broken, fixed = broken.to(device), fixed.to(device)

        outputs = model(broken)
        loss = criterion(outputs, fixed)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for broken, fixed in tqdm(loader, desc='Validation', leave=False):
            broken, fixed = broken.to(device), fixed.to(device)
            outputs = model(broken)
            loss = criterion(outputs, fixed)
            total_loss += loss.item()

    return total_loss / len(loader)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train UNet to fix line art')
    parser.add_argument('--train_dir', type=str, default='train', help='Training data directory')
    parser.add_argument('--val_dir', type=str, default='val', help='Validation data directory')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--visualize_every', type=int, default=0, help='Visualize every N epochs (0 to disable)')
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Data
    train_dataset = LineArtDataset(args.train_dir)
    val_dataset = LineArtDataset(args.val_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Model & training setup
    model = UNet(n_channels=1, n_classes=1, bilinear=True).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = CombinedLoss(bce_weight=0.5, l1_weight=0.2, l2_weight=0.3)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=args.lr * 1e-3)

    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print('-' * 50)

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"LR:         {current_lr:.2e}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(args.checkpoint_dir, 'best_model.pth'))
        else:
            patience_counter += 1

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth'))

            if args.visualize_every and ((epoch + 1) % args.visualize_every == 0):
                try:
                    visualize_batch(epoch, model, val_loader, device, num_samples=4, results_dir=args.results_dir)
                    plot_training_history(epoch, history, results_dir=args.results_dir)
                except Exception as e:
                    print(f"Visualization failed: {e}")

        # Early stopping
        if patience_counter >= 5:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Save training history
    with open(os.path.join(args.results_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + '=' * 50)
    print('Training complete!')
    print(f'Best Validation Loss: {best_val_loss:.4f}')
    


if __name__ == '__main__':
    # Ensure DataLoader workers are spawned under main on Windows
    main()
