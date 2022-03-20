from osgeo import gdal
import numpy as np

def convert_label(label, inverse=False):
    label_mapping = {
        0: 0,
        60: 1
    }
    tmp = label.copy()
    if inverse:
        for v, k in label_mapping.items():
            label[tmp == k] = v
    else:
        for k, v in label_mapping.items():
            label[tmp == k] = v
        label[label > len(label_mapping) - 1] = 0
    return label


def read_image(filename):
    dataset = gdal.Open(filename)
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    im_proj = dataset.GetProjection()  # 地图投影信息
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵
    del dataset

    if len(im_data) == 3:
        return im_data.transpose([1,2,0])
    else:
        return im_proj, im_geotrans, im_data.transpose([1,2,0])
import os
from collections import OrderedDict

import torch
import torch.utils.data as data
from PIL import Image
import glob
import cv2
from tqdm import tqdm, trange

COLOR_MAP = OrderedDict(
    Background=(0, 0, 0),
    Building=(255, 255, 255),

)


LABEL_MAP = OrderedDict(
    Background=0,
    Building=1,
)



def reclassify(cls):
    cls_mtx = np.array(cls)
    new_cls = np.zeros((cls_mtx.shape[0],cls_mtx.shape[1]))
    for idx, label in enumerate(COLOR_MAP.values()):
        new_cls = np.where(cls == label, np.ones_like(new_cls)*idx, new_cls)
    return new_cls
class DataSet(data.Dataset):
    def __init__(self, data_root, transforms=None):
        super(DataSet, self).__init__()
        root = os.path.join(data_root)
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        image_dir = os.path.join(root, 'image')
        mask_dir = os.path.join(root, 'mask')

        # txt_path = os.path.join(root, "ImageSets", "Segmentation", txt_name)
        # assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        # with open(os.path.join(txt_path), "r") as f:
        #     file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        self.images = sorted(glob.glob(os.path.join(image_dir , "*.tif")))
        self.masks = sorted(glob.glob(os.path.join(mask_dir , "*.tif")))
        assert (len(self.images) == len(self.masks))
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        # img = cv2.imread(self.images[index], flags=cv2.IMREAD_UNCHANGED)
        # target = cv2.imread(self.masks[index], flags=cv2.IMREAD_UNCHANGED)
        img = read_image(self.images[index])
        target = cv2.imread(self.masks[index])

        if self.transforms is not None:
            img,target= self.transforms(img,target)
            # target=torch.LongTensor(target)
        return img, target
    def __len__(self):
        return len(self.images)
    def __call__(self, train=True,*args, **kwargs):
        X = []
        y = []
        for i in trange(int(len(self.images)/128) if train == True else int(len(self.images)/8)):
            X.append(read_image(self.images[i])[2].reshape((-1, 6)))
        for i in trange(int(len(self.masks)/128) if train == True else int(len(self.masks)/8)):
            y.append(cv2.imread(self.masks[i],flags=cv2.IMREAD_UNCHANGED).flatten())
        X = np.array(X)
        y = np.array(y)
        y[y == 10] = 1
        y[y == 20] = 2
        y[y == 30] = 3
        y[y == 40] = 4
        y[y >=5 ] = 0
        X = X.reshape((-1,6))
        y = y.reshape((-1))

        return X,y
    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

if __name__ == '__main__':
    d = read_image('D:/Dataset/SemanticSegmentation/Landsat30/512_128/train/image/0.tif')
    data = DataSet('D:/Dataset/SemanticSegmentation/Landsat30/512_128/train/image/0.tif',transforms=None)


    print(data)