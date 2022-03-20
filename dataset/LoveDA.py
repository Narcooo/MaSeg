import os
from collections import OrderedDict

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import glob
import cv2

COLOR_MAP = OrderedDict(
    Background=(255, 255, 255),
    Building=(255, 0, 0),
    Road=(255, 255, 0),
    Water=(0, 0, 255),
    Barren=(159, 129, 183),
    Forest=(0, 255, 0),
    Agricultural=(255, 195, 128),
)


LABEL_MAP = OrderedDict(
    Background=0,
    Building=1,
    Road=2,
    Water=3,
    Barren=4,
    Forest=5,
    Agricultural=6
)



def reclassify(cls):
    new_cls = np.ones_like(cls, dtype=np.int64) * -1
    for idx, label in enumerate(LABEL_MAP.values()):
        new_cls = np.where(cls == idx, np.ones_like(cls)*label, new_cls)
    return new_cls
class DataSet(data.Dataset):
    def __init__(self, data_root, transforms=None):
        super(DataSet, self).__init__()
        #assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        root = os.path.join(data_root)
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        image_dir = os.path.join(root, 'Rural/images_png')
        mask_dir = os.path.join(root, 'Rural/masks_png')

        self.images = sorted(glob.glob(os.path.join(image_dir , "*.png")))
        self.masks = sorted(glob.glob(os.path.join(mask_dir , "*.png")))
        assert (len(self.images) == len(self.masks))
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = cv2.imread(self.images[index], flags=cv2.IMREAD_UNCHANGED)
        target = cv2.imread(self.masks[index], flags=cv2.IMREAD_UNCHANGED)



        if self.transforms is not None:
            img,target= self.transforms(img,target)

            # a=np.array(target)
            # b=a-1
            # c = np.max(a)
            # d = np.min(a)
            # e = np.max(b)
            # f = np.min(b)
            target=target-1
            # target=torch.LongTensor(target)
        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=0)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
