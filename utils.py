import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm
from osgeo import gdal

import matplotlib.pyplot as plt

dataset = gdal.Open("D:/Dataset/SemanticSegmentation/AerialImageDataset/train/images/austin1.tif")
l = [[],[],[]]

