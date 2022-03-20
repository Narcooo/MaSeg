import os
import numpy as np
from PIL import Image

import glob
num_classes = 8
root = os.path.join("D:/Dataset/SemanticSegmentation/2021LoveDA/Train/Rural/")
mask_dir = os.path.join(root, 'masks_png')
masks = sorted(glob.glob(os.path.join(mask_dir , "*.png")))
num_l = []
image_l = []
i_l = []
for i in range(0,num_classes):
    num_l.append(0)
for i in range(0,len(masks)):
    image_l.append(False)
for i in range(0,len(masks)):
    target = Image.open(masks[i])
    img = np.array(target)
    if img.min == 0:
        i_l.append(masks[i])
    # for j in range(0,img.shape[0]):
    #     for k in range(0,img.shape[1]):
    #         # num_l[img[j][k]] += 1
    #         if img[j][k] == 0:
    #             image_l[i] = True
    #             print(i,j,k)



if __name__ == '__main__':
    print(image_l)