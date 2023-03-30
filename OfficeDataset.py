import os
import pandas as pd
from torch.utils.data import Dataset
# from torchvision.io import read_image
from PIL import Image


class OfficeDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, strong_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.strong_transform = strong_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        with open(img_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
            label = self.img_labels.iloc[idx, 1]
            image_transformed = None
            image_transformed2 = None
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            if self.strong_transform:
                image_transformed = self.strong_transform(image)
                image_transformed2 = self.strong_transform(image)
            return image, label, image_transformed, image_transformed2
