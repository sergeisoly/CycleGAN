import glob
import os
import shutil
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms



class ImageDataset(Dataset):
    """ 
    https://github.com/aitorzip/PyTorch-CycleGAN/blob/master/datasets.py
    trainA, trainB not train/A, train/B

    """
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, f"{mode}A") + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, f"{mode}B") + '/*.*'))

    def __getitem__(self, index):
        img_A = Image.open(self.files_A[index % len(self.files_A)])
        item_A = self.transform(img_A)

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))

        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

def prepare_dataset(data_path):
    for data in ['A', 'B']:
        files = glob.glob(os.path.join('data/', f"train{data}") + '/*.*')

        # remove gray images
        for img in files:
            img_ = Image.open(img)
            img_ = transforms.ToTensor()(img_)
            if img_.shape[0] == 1:
                os.remove(img)
                files.remove(img)
        # create test data
        files_test = files[:10]
        for img in files_test:
            shutil.copy(img, f"{data_path}test{data}")
            os.remove(img)