# -*- coding: utf-8 -*-

import torch.utils.data as data
import os
from PIL import Image
from torchvision import transforms

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def load_img(filepath):
    img = Image.open(filepath)
    return img


def input_transform(crop_size):
    return transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.ToTensor(),
    ])


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, crop_size):
        super(DatasetFromFolder, self).__init__()
        self.input_transform = input_transform(crop_size)
        self.image_filenames = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if is_image_file(x)]     # 这种只适合一个文件夹内全是图片的，子文件夹内图片不会读取
        self.secret_filenames = self.image_filenames[:len(self.image_filenames)//2]
        self.cover_filenames = self.image_filenames[len(self.image_filenames)//2:]
    def __getitem__(self, index):
        secret = load_img(self.secret_filenames[index])
        cover = load_img(self.cover_filenames[index])
        if self.input_transform:
            secret = self.input_transform(secret)
            cover = self.input_transform(cover)

        return secret, cover

    def __len__(self):
        return len(self.secret_filenames)
