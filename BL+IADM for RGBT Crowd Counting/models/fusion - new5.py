import torch.nn as nn
import torch
from torch.nn import functional as F

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        
        self.b1 = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.b3 = nn.Sequential(
            nn.Conv2d(128,256,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.b4 = nn.Sequential(
            nn.Conv2d(256,512,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.b5 = nn.Sequential(
            nn.Conv2d(512+512,512,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1),
            nn.ReLU(inplace=True)
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
        RGB = self.b4(RGB)
        T = self.b4(T)
        x = torch.cat([RGB,T],1)
        x = self.b5(x)
        x = F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=True)
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