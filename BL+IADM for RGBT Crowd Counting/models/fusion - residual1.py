import torch.nn as nn
import torch
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.shortcut = nn.Sequential()

        if in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
            )
                

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()

        self.layer1 = nn.Sequential(
            ResidualBlock(3,64),
            ResidualBlock(64,64),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64,128),
            ResidualBlock(128,128),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128,256),
            ResidualBlock(256,256),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            ResidualBlock(256,512),
            ResidualBlock(512,512)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.layer6 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )
        self.shortcut = nn.Conv2d(512, 128, kernel_size=1, bias=False)


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
        x2 = self.layer5(x)
        x2 += self.shortcut(x)
        x2 = self.layer6(x2)

        return torch.abs(x2)

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