import torch.nn as nn
import torch
from torch.nn import functional as F

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, \
                stride=1, NL='relu', same_padding=False, bn=False, dilation=1):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0
        self.conv = []
        if dilation==1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else nn.Identity()
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class FusionModel(nn.Module):
    def __init__(self, ratio=0.6):
        super(FusionModel, self).__init__()
        c1 = int(64 * ratio)
        c2 = int(128 * ratio)
        c3 = int(256 * ratio)
        c4 = int(512 * ratio)
        
        self.b1 = nn.Sequential(
            Conv2d(3, 64, 3, same_padding=True, NL='relu'),
            Conv2d(64, 64, 3, same_padding=True, NL='relu'),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.b2 = nn.Sequential(
            Conv2d(64, 128, 3, same_padding=True, NL='relu'),
            Conv2d(128, 128, 3, same_padding=True, NL='relu'),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.b3 = nn.Sequential(
            Conv2d(128, 256, 3, same_padding=True, NL='relu'),
            Conv2d(256, 256, 3, same_padding=True, NL='relu'),
            Conv2d(256, 256, 3, same_padding=True, NL='relu'),
            Conv2d(256, 256, 3, same_padding=True, NL='relu'),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.b4 = nn.Sequential(
            Conv2d(256+256, 256, 1, same_padding=True, NL='relu'),
            Conv2d(256, 512, 3, same_padding=True, NL='relu'),
            Conv2d(512, 512, 3, same_padding=True, NL='relu'),
            Conv2d(512, 512, 3, same_padding=True, NL='relu'),
            Conv2d(512, 512, 3, same_padding=True, NL='relu')
        )

        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )
        self._initialize_weights()

    def forward(self, RGBT):
        RGB = RGBT[0]
        T = RGBT[1]
        RGB = self.b1(RGB)
        T = self.b1(T)
        RGB = self.b2(RGB)
        T = self.b2(T)
        RGB = self.b3(RGB)
        T = self.b3(T)
        x = torch.cat([RGB,T],1)
        x = self.b4(x)

        #x = F.upsample_bilinear(x, scale_factor=2)
        x = self.reg_layer(x)
        return torch.abs(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, std=0.01)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def fusion_model():
    model = FusionModel()
    return model
