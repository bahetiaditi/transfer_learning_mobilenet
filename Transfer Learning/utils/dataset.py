# utils/dataset.py
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class ISICDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None ,mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]


        if '_aug_' in img_name:

            mask_name = img_name.replace('image_', 'mask_')
        else:

            base_filename = img_name.split('.')[0]
            mask_name = f'ISIC_{base_filename.split("_")[-1]}.png'


        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name )


        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.image_transform:
          image = self.image_transform(image)
        if self.mask_transform:
          mask = self.mask_transform(mask)

        return image, mask
