import os
import time
import json

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from segmentors import Segmentor
from dataset.LoveDA import COLOR_MAP
def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    aux = False  # inference time not need aux_classifier
    classes = 20
    weights_path = "weights/model_49.pth"
    img_path = "D:/Dataset/SemanticSegmentation/2021LoveDA/Test/Rural/images_png/4192.png"
    pallette_dict = COLOR_MAP
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    pallette = []
    for v in pallette_dict.values():
        pallette+=v

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = Segmentor(num_classes = 7)

    # delete weights about aux_classifier
    weights_dict = torch.load(weights_path, map_location='cpu')['model']
    for k in list(weights_dict.keys()):
        if "aux" in k:
            del weights_dict[k]

    # load weights
    model.load_state_dict(weights_dict)
    model.to(device)

    # load image
    original_img = Image.open(img_path)

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                              std=(0.229, 0.224, 0.225))])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))
        a = output.softmax(dim=1)
        b = a.argmax(dim=1).squeeze(0)
        b = b.to("cpu").numpy().astype(np.uint8)
        prediction = output.argmax(dim=1)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        mask = Image.fromarray(b)
        mask.putpalette(pallette)
        mask.save("test_result.png")


if __name__ == '__main__':
    main()
