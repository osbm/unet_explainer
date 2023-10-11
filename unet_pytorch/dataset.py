import torch
from pathlib import Path
from PIL import Image
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ProstateDataset(torch.utils.data.Dataset):
    def __init__(self, folder: str, transform=None):
        self.folder = Path(folder)
        self.transform = transform

        self.images = sorted(os.listdir(self.folder / "image"))
        self.masks = sorted(os.listdir(self.folder / "mask"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.folder / "image" / self.images[idx]
        mask_path = self.folder / "mask" / self.masks[idx]

        image = np.array(Image.open(str(image_path)).convert("L"))
        mask = np.array(Image.open(str(mask_path)).convert("L"))

        if self.transform is not None:
            result = self.transform(image=image, mask=mask)
            image = result["image"]
            mask = result["mask"]
        else:
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)

        # mask values are 0, 1, 2 but in this image we have 0, 128, 255 lets fix this
        mask[mask == 128] = 1
        mask[mask == 255] = 2
        mask = mask.unsqueeze(0)

        image = image.float() / 255.0
        # mask = mask.float() / 2.0

        return image, mask


def get_transforms(image_size=512):
    train_transforms = A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # random rotate 5 degrees
        A.Rotate(limit=5, p=0.5),
        # random offset 10 pixels
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5),
        # random brightness and contrast
        A.RandomBrightnessContrast(p=0.5),
        # A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])

    valid_transforms = A.Compose([
        A.Resize(height=image_size, width=image_size),
        # A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])

    return train_transforms, valid_transforms
