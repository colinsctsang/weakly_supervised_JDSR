from os.path import join
from torchvision.transforms import Compose, CenterCrop, ToTensor, Scale, RandomCrop, Resize
from dataset import DatasetFromFolder, TestingDatasetFromFolder

crop_size = 128
test_crop_size = 128


def get_w2s(dest="dataset"):
    output_image_dir = join(dest, "w2s_avg1")
    return output_image_dir



def HR_2_transform(crop_size, scale):
    return Compose([
        Resize(crop_size // scale),
        ToTensor(),
    ])


def HR_4_transform(crop_size):
    return Compose([
        RandomCrop((crop_size, crop_size)),
    ])

def HR_4_test_transform(test_crop_size):
    return Compose([
        CenterCrop((test_crop_size, test_crop_size)),
    ])


def get_training_set(scale):
    root_dir = get_w2s()
    train_dir = join(root_dir, "train")

    return DatasetFromFolder(train_dir,
                             HR_2_transform=HR_2_transform(crop_size, scale),
                             HR_4_transform=HR_4_transform(crop_size))


def get_test_set(scale):
    gt_dir = join("dataset", "w2s_avg400", "test")
    root_dir = get_w2s()
    test_dir = join(root_dir, "test")

    return TestingDatasetFromFolder(test_dir, gt_dir,
                                    HR_2_transform=HR_2_transform(test_crop_size, scale),
                                    HR_4_transform=HR_4_test_transform(test_crop_size),
                                    HR_avg400_transform=HR_4_test_transform(test_crop_size))


