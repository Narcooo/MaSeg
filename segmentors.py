import torch
import math
from torch import nn, Tensor
from torch.nn import functional as F
from backbone.swintransformer import swin_small_patch4_window7_224
from head.upernet import UPerHead
from backbone.hrnet.hrnet import hrnet
from head.segformerhead import SegFormerHead
from backbone.mit import MiT


class Segmentor(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.backbone = swin_small_patch4_window7_224(num_classes=num_classes)
        #self.backbone = swin_base_patch4_window8_1024()
        # self.num_classes = num_classes
        # self.backbone = hrnet(pretrained=False, version='48')
        self.decode_head = UPerHead(self.backbone.num_feature, num_classes=num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        y = self.decode_head(y)   # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
        return y


if __name__ == '__main__':
    model = Segmentor(7)
    #model.load_state_dict(torch.load('checkpoints/pretrained/segformer/segformer.b0.ade.pth', map_location='cpu'))
    x = torch.zeros(3, 3, 1024, 1024)
    y = model(x)
    print(y.shape)