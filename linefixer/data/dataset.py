from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class LineArtDataset(Dataset):
    """Dataset for broken/fixed line art pairs"""
    def __init__(self, root_dir, transform=None, image_size=128):
        """
        Args:
            root_dir: Directory with 'broken' and 'fixed'
            transform: Optional transform to be applied
            image_size: Size to resize images to
        """
        self.root_dir = Path(root_dir)
        self.broken_dir = self.root_dir / 'broken'
        self.fixed_dir = self.root_dir / 'fixed'
        self.transform = transform
        self.image_size = image_size

        # Get list of images
        self.images = sorted([f.name for f in self.broken_dir.glob('*.png')])
        print(f'Found {len(self.images)} image pairs in {root_dir}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        # Load images
        broken_path = self.broken_dir / img_name
        fixed_path = self.fixed_dir / img_name

        broken = Image.open(broken_path).convert('L')
        fixed = Image.open(fixed_path).convert('L')

        # Resize if needed
        if broken.size != (self.image_size, self.image_size):
            broken = broken.resize((self.image_size, self.image_size), Image.BILINEAR)
            fixed = fixed.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Convert to tensors
        broken = torch.FloatTensor(np.array(broken)) / 255.0
        fixed = torch.FloatTensor(np.array(fixed)) / 255.0

        # Add channel dimension
        broken = broken.unsqueeze(0)
        fixed = fixed.unsqueeze(0)

        # Apply transforms if any
        if self.transform:
            # Apply same transform to both images
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            broken = self.transform(broken)
            torch.manual_seed(seed)
            fixed = self.transform(fixed)

        return broken, fixed
