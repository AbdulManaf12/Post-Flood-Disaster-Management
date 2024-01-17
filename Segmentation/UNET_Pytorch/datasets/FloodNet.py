import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

class FloodNetDataset(Dataset):
    def __init__(self, root_directory, split='train', transform=None, IMAGE_SHAPE=(128, 128, 3)):
        self.root = root_directory
        self.split = split
        self.transform = transform
        self.SIZE_X = IMAGE_SHAPE[0]
        self.SIZE_Y = IMAGE_SHAPE[1]
        self.CHANNELS = IMAGE_SHAPE[2]

        self.images_directory = os.path.join(self.root, split, f"{split}-org-img")

        self.filenames = os.listdir(self.images_directory)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.split, f"{self.split}-org-img", self.filenames[idx])
        image, mask = self.load_image_and_mask(image_path, self.SIZE_X, self.SIZE_Y)

        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image_and_mask(self, image_path, size_x, size_y):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (size_x, size_y))

        mask_path = (image_path.replace('org', 'label')).replace('jpg', 'png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (size_x, size_y))

        return img, mask
