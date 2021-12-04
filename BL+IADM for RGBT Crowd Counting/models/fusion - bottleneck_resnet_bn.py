import torch.nn as nn
import torch
from torch.nn import functional as F

class BottleNeck(nn.Module):
    mul = 4
    def __init__(self, in_planes, out_planes, stride=1):
        super(BottleNeck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        
        self.conv3 = nn.Conv2d(out_planes, out_planes*self.mul, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes*self.mul)
        
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_planes != out_planes*self.mul:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes*self.mul, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes*self.mul)
            )
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            BottleNeck(64,64,1),
            BottleNeck(256,64,1),
            BottleNeck(256,64,2)
        )
        self.layer3 = nn.Sequential(
            BottleNeck(256,128,1),
            BottleNeck(512,128,1),
            BottleNeck(512,128,1),
            BottleNeck(512,128,2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            BottleNeck(512,256,1),
            BottleNeck(1024,256,1),
            BottleNeck(1024,256,1),
            BottleNeck(1024,256,1),
            BottleNeck(1024,256,1),
            BottleNeck(1024,256,2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

        self._initialize_weights()

    def forward(self, RGBT):
        RGB = RGBT[0]
        T = RGBT[1]
        RGB = self.layer1(RGB)
        T = self.layer1(T)
        RGB = self.layer2(RGB)
        T = self.layer2(T)
        RGB = self.layer3(RGB)
        T = self.layer3(T)
        x = torch.cat([RGB,T],1)
        x = self.layer4(x)
        x = self.layer5(x)

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