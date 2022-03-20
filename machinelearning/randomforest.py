import numpy as np
import sklearn
from osgeo import gdal
import os
import argparse
import time
import datetime
import transforms
from sklearn.ensemble import RandomForestClassifier
# from dataset.AIRS import DataSet
from dataset.Landsat30 import DataSet
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from segmentors import Segmentor
import pickle
import cv2

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

def main(args):
    results_file = "results/results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    rfc = RandomForestClassifier(n_estimators=50,max_depth=100, n_jobs=1)
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    #train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    train_images_path = args.data_path + 'Train/'
    test_images_path = args.data_path + 'Test/'
    img_size = 512
    data_transform = {
        "train": transforms.Compose([
                                     # T.RandomCrop(img_size,fill=1),
                                     # transforms.ToTensor(),
                                     transforms.RandomCrop(img_size, mask_background_fill=0),
                                     transforms.RandomHorizontalFlip(),

                                     transforms.Normalize(mean=(123.675, 116.28, 103.53),
                           std=(58.395, 57.12, 57.375))
        ]),
                                     # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
        "val": transforms.Compose([
                                   # transforms.ToTensor(),
                                   transforms.CenterCrop(512),

                                   transforms.Normalize(mean=(123.675, 116.28, 103.53),
                                 std=(58.395, 57.12, 57.375))
        ])}
                                   # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])}

    # 实例化训练数据集
    # train_dataset = DataSet(data_root=train_images_path,
    #                           transforms=None)
    # X_train,y_train = train_dataset()
    # rfc.fit(X_train, y_train)
    # file = open('D:/pypro/SwinTransformer/machinelearning/weights/weight/rfc', "wb")
    # # 将模型写入文件：
    # pickle.dump(rfc, file)
    # # 最后关闭文件：
    # file.close()
    def read_image(filename):
        dataset = gdal.Open(filename)
        im_width = dataset.RasterXSize  # 栅格矩阵的列数
        im_height = dataset.RasterYSize  # 栅格矩阵的行数
        im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
        im_proj = dataset.GetProjection()  # 地图投影信息
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵
        del dataset
        if len(im_data) == 3:
            return im_data.transpose([1, 2, 0])
        else:
            return im_proj, im_geotrans, im_data.transpose([1, 2, 0])
    # # 实例化验证数据集
    # test_dataset = DataSet(data_root=test_images_path,
    #                         transforms=None)
    # X_test, y_test = test_dataset(train=False)
    # score = rfc.score(X_test, y_test)
    # y_pred = rfc.predict(X_test)
    # f1_micro = f1_score(y_test,y_pred,average='micro')
    # f1_macro = f1_score(y_test, y_pred, average='macro')
    # # confusion_matrix = confusion_matrix(y_test, y_pred)
    # # 分类准确率 accuracy
    # accuracy = accuracy_score(y_test, y_pred)
    # # 精确率Precision
    # precision_micro = precision_score(y_test, y_pred,average='micro')
    # precision_macro = precision_score(y_test, y_pred, average='macro')
    # # 召回率 recall
    # recall_micro = recall_score(y_test, y_pred,average='micro')
    # recall_macro = recall_score(y_test, y_pred, average='macro')
    # print(score,f1_micro,f1_macro,accuracy,precision_micro,precision_macro,recall_micro,recall_macro)
    X_test = read_image('D:/Dataset/SemanticSegmentation/Landsat30/origin/test_image/p123_r038_l8_20170726.img')[2].reshape((-1, 6))
    # X_test = read_image('D:/Dataset/SemanticSegmentation/Landsat30/origin/test_image/p123_r038_l8_20170726.img')[2]
    # X_test = read_image('D:/Dataset/SemanticSegmentation/Landsat30/512_128/train/image/704.tif')[
    #     2].reshape((-1, 6))
    X_test = X_test.astype(np.float64)
    X_test = X_test*255/65536
    X_test = X_test.astype(np.uint8)
    with open('D:/pypro/SwinTransformer/machinelearning/weights/weight/rfc', 'rb') as f:
        rfc = pickle.load(f)
    y_test = rfc.predict(X_test)
    y_test = y_test.reshape((5535,4674))
    # y_test = y_test * 10
    y_test[y_test==10]=np.array(250,160,255)
    y_test[y_test == 20] = np.array(0, 100, 0)
    y_test[y_test == 30] = np.array(100, 255, 0)
    y_test[y_test == 40] = np.array(0, 255, 120)
    y_test[y_test == 0] = np.array(0, 0, 0)
    cv2.imwrite('result.tif',y_test)



        # torch.save(model.state_dict(), "weights/"+args.save_path +"model-{}.pth".format(epoch))

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', type=str,
                        default="D:/Dataset/SemanticSegmentation/Landsat30/512_128/")


    opt = parser.parse_args()

    main(opt)
