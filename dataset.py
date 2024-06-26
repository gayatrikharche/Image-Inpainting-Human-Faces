import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image


# Custom Dataset to load images
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_images=5000):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.num_images = num_images
        for subdir, _, files in os.walk(self.root_dir):
            for file in files:
                self.images.append(os.path.join(subdir, file))
        self.images = self.images[:self.num_images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image