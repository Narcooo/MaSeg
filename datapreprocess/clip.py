import numpy as np
from PIL import Image
import glob
import os
import torch
import cv2
from tqdm import tqdm, trange
from osgeo import gdal
import re
# image_settings = ['jpg','tif','png','jpeg','tiff']
class DatasetImageCropper(object):
    def __init__(self):
        self.image_settings = ['jpg', 'tif', 'png', 'jpeg', 'tiff']
        self.clip_h_iter = 0
        self.clip_w_iter = 0
        self.clip_h = 0
        self.clip_w = 0
        self.split_num = []
    def __call__(self,
                 data_path,
                 output_path,
                 w_size,
                 h_size,
                 image_type,
                 padding_fill=True,
                 is_geotiff_image=False,
                 key=lambda name:int(name.split('_')[-2])):
        assert image_type in self.image_settings, f".name should be in {self.image_settings}"
        isExists = os.path.exists(output_path)
        if not isExists:
            os.makedirs(output_path)
        image_num = 0
        total_num = 0
        images = sorted(glob.glob(os.path.join(data_path , "*."+image_type)),key=key)
        out_images = output_path + "/"

        for index in trange(0, len(images)):
            img = cv2.imread(images[index],flags=cv2.IMREAD_UNCHANGED)
            path = images[index].split('\\')[-1].split('.')[0]
            self.clip_h = img.shape[0] % h_size
            self.clip_w = img.shape[1] % w_size
            self.clip_h_iter = img.shape[0] // h_size + 1 if self.clip_h != 0 else img.shape[0] // h_size
            self.clip_w_iter = img.shape[1] // w_size + 1 if self.clip_w != 0 else img.shape[1] // w_size
            if padding_fill == True:
                if self.clip_h != 0:
                    padding_mtx = np.zeros((h_size - self.clip_h,img.shape[1],
                                            img.shape[2])) if len(img.shape) == 3 else np.zeros((h_size - self.clip_h,img.shape[1]))
                    img = np.concatenate((img,padding_mtx),axis=0).astype('uint8')
                if self.clip_w != 0:
                    padding_mtx = np.zeros((img.shape[0],w_size - self.clip_w,
                                            img.shape[2])) if len(img.shape) == 3 else np.zeros((img.shape[0],w_size - self.clip_w))
                    img = np.concatenate((img,padding_mtx),axis=1).astype('uint8')
                for k in range(0,self.clip_h_iter):
                    for j in range(0,self.clip_w_iter):
                        img_crop = img[k * h_size:(k + 1) * h_size,
                                       j * w_size:(j + 1) * w_size,]
                        image_num +=1
                        total_num +=1
                        cv2.imwrite(out_images + path + "_"+str(image_num) + "_"+str(total_num) + "." + image_type,img_crop)
                self.split_num.append(image_num)
                image_num = 0
            else:
                for k in range(0,self.clip_h_iter if self.clip_h == 0 else self.clip_h_iter - 1):
                    for j in range(0,self.clip_w_iter  if self.clip_w == 0 else self.clip_w_iter - 1):
                        img_crop = img[k * h_size:(k + 1) * h_size,
                                       j * w_size:(j + 1) * w_size,]
                        image_num +=1
                        total_num +=1
                        cv2.imwrite(out_images + path + "_"+str(image_num) + "_"+str(total_num) + "." + image_type,img_crop)
                    if self.clip_w != 0:
                        img_crop = img[k * h_size:(k + 1) * h_size,
                                       (self.clip_w_iter - 1) * w_size:(self.clip_w_iter - 1) * w_size + self.clip_w,]
                        image_num += 1
                        total_num += 1
                        cv2.imwrite(out_images + path + "_"+str(image_num) + "_"+str(total_num) + "." + image_type,img_crop)
                if self.clip_h != 0:
                    for j in range(0,self.clip_w_iter - 1):
                        img_crop = img[(self.clip_h_iter - 1) * h_size:(self.clip_h_iter - 1) * h_size + self.clip_h,
                                       j * w_size:(j + 1) * w_size,]
                        image_num += 1
                        total_num += 1
                        cv2.imwrite(out_images + path + "_"+str(image_num) + "_"+str(total_num) + "." + image_type,img_crop)
                if self.clip_h != 0 and self.clip_w != 0:
                    img_crop = img[(self.clip_h_iter - 1) * h_size:(self.clip_h_iter - 1) * h_size + self.clip_h,
                                   (self.clip_w_iter - 1) * w_size:(self.clip_w_iter - 1) * w_size + self.clip_w,]
                    image_num += 1
                    total_num += 1
                    cv2.imwrite(out_images + path + "_"+str(image_num) + "_"+str(total_num) + "." + image_type,img_crop)
                self.split_num.append(image_num)
                image_num = 0

        with open(output_path.split('/')[-1] + '.txt', "w") as f:
            for i in range(len(self.split_num)):
                f.write(str(self.split_num[i]))
                f.write('\n')
        self.split_num=[]
        return 0

if __name__ == '__main__':
    cropper = DatasetImageCropper()
    # cropper(data_path="D:/Dataset/SemanticSegmentation/AerialImageDataset/train/images",
    #         output_path="D:/Dataset/SemanticSegmentation/AerialImageDataset_512/train/images",
    #         w_size=512,
    #         h_size=512,
    #         image_type='tif',
    #         padding_fill=True,
    #         key=lambda name:int(re.findall(r'\d',name)[0]))
    cropper(data_path="D:/Dataset/SemanticSegmentation/AerialImageDataset/train/gt",
            output_path="D:/Dataset/SemanticSegmentation/AerialImageDataset_512/train/gt",
            w_size=512,
            h_size=512,
            image_type='tif',
            padding_fill=True,
            key=lambda name:int(re.findall(r'\d',name)[0]))
    cropper(data_path="D:/Dataset/SemanticSegmentation/AerialImageDataset/val/images",
            output_path="D:/Dataset/SemanticSegmentation/AerialImageDataset_512/val/images",
            w_size=512,
            h_size=512,
            image_type='tif',
            padding_fill=True,
            key=lambda name:int(re.findall(r'\d',name)[0]))
    cropper(data_path="D:/Dataset/SemanticSegmentation/AerialImageDataset/val/gt",
            output_path="D:/Dataset/SemanticSegmentation/AerialImageDataset_512/val/gt",
            w_size=512,
            h_size=512,
            image_type='tif',
            padding_fill=True,
            key=lambda name:int(re.findall(r'\d',name)[0]))
    # cropper(data_path="D:/Dataset/SemanticSegmentation/AIRS/trainval/train/label",
    #         output_path="D:/Dataset/SemanticSegmentation/AIRS_512/train/label",
    #         w_size=512,
    #         h_size=512,
    #         image_type='tif',
    #         padding_fill=True,
    #         key=lambda  name:int(name.split('_')[-1].split('.')[0]))
    # cropper(data_path="D:/Dataset/SemanticSegmentation/AIRS/trainval/train/image",
    #         output_path="D:/Dataset/SemanticSegmentation/AIRS_512/train/image",
    #         w_size=512,
    #         h_size=512,
    #         image_type='tif',
    #         padding_fill=True,
    #         key=lambda  name:int(name.split('_')[-1].split('.')[0]))
    # cropper(data_path="D:/Dataset/SemanticSegmentation/AIRS/trainval/val/label_vis",
    #         output_path="D:/Dataset/SemanticSegmentation/AIRS_512/val/label_vis",
    #         w_size=512,
    #         h_size=512,
    #         image_type='tif',
    #         padding_fill=True,
    #         key=lambda name:int(name.split('_')[-2]))
    # cropper(data_path="D:/Dataset/SemanticSegmentation/AIRS/trainval/val/label",
    #         output_path="D:/Dataset/SemanticSegmentation/AIRS_512/val/label",
    #         w_size=512,
    #         h_size=512,
    #         image_type='tif',
    #         padding_fill=True,
    #         key=lambda  name:int(name.split('_')[-1].split('.')[0]))
    # cropper(data_path="D:/Dataset/SemanticSegmentation/AIRS/trainval/val/image",
    #         output_path="D:/Dataset/SemanticSegmentation/AIRS_512/val/image",
    #         w_size=512,
    #         h_size=512,
    #         image_type='tif',
    #         padding_fill=True,
    #         key=lambda  name:int(name.split('_')[-1].split('.')[0]))