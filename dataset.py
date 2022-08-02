import torch.utils.data as data
import torchvision
from os import listdir
from os.path import join
from PIL import Image
import numpy as np

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()

    randint = np.random.randint(0, 4)
    if randint == 0:
        y = y.rotate(90)
    elif randint == 1:
        y = y.rotate(180)
    elif randint ==2:
        y = y.rotate(270)
    else:
        pass
    return y

def load_test_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, HR_2_transform=None, HR_4_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.HR_2_transform = HR_2_transform
        self.HR_4_transform = HR_4_transform


    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        HR_4 = self.HR_4_transform(input)
        HR_2 = self.HR_2_transform(HR_4)
        to_tensor = torchvision.transforms.ToTensor()
        HR_4 = to_tensor(HR_4)
        return HR_2, HR_2, HR_4

    def __len__(self):
        return len(self.image_filenames)

class TestingDatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, gt_dir, HR_2_transform=None, HR_4_transform=None, HR_avg400_transform=None):
        super(TestingDatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.image_gt_filenames = [join(gt_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.HR_2_transform = HR_2_transform
        self.HR_4_transform = HR_4_transform
        self.HR_avg400_transform = HR_avg400_transform


    def __getitem__(self, index):
        # get noise-free image
        input_gt = load_test_img(self.image_gt_filenames[index])
        HR_avg400 = self.HR_avg400_transform(input_gt)
        # get noisy image
        input = load_test_img(self.image_filenames[index])
        HR_4 = self.HR_4_transform(input)
        # generate LR images
        HR_2 = self.HR_2_transform(HR_4)
        to_tensor = torchvision.transforms.ToTensor()
        HR_4 = to_tensor(HR_4)
        HR_avg400 = to_tensor(HR_avg400)
        return HR_2, HR_4, HR_avg400

    def __len__(self):
        return len(self.image_filenames)


