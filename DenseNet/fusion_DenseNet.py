import torch.nn as nn
import torch
from torch.nn import functional as F

class DenseNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.num_classes = num_classes
        self.growth_rate = 32
        self.base_feature = nn.Sequential(nn.Conv2d(6, 64, 7, stride=2, padding=3, bias=False),
                                          nn.BatchNorm2d(64),
                                          nn.ReLU(inplace=True),
                                          nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                          )

        self.dense_layer1 = nn.Sequential(nn.BatchNorm2d(64),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(64, self.growth_rate * 4, 1, bias=False),

                                          nn.BatchNorm2d(self.growth_rate * 4),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(self.growth_rate * 4, self.growth_rate, 3, padding=1, bias=False),
                                          )

        self.dense_layer2 = nn.Sequential(nn.BatchNorm2d(96),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(96, 128, 1, bias=False),

                                          nn.BatchNorm2d(128),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                          )

        self.dense_layer3 = nn.Sequential(nn.BatchNorm2d(128),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, 128, 1, bias=False),

                                          nn.BatchNorm2d(128),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                          )

        self.dense_layer4 = nn.Sequential(nn.BatchNorm2d(160),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(160, 128, 1, bias=False),

                                          nn.BatchNorm2d(128),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                          )

        self.dense_layer5 = nn.Sequential(nn.BatchNorm2d(192),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(192, 128, 1, bias=False),

                                          nn.BatchNorm2d(128),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                          )

        self.dense_layer6 = nn.Sequential(nn.BatchNorm2d(224),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(224, 128, 1, bias=False),

                                          nn.BatchNorm2d(128),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                          )

        self.transition1 = nn.Sequential(nn.BatchNorm2d(256),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(256, 1, 1, bias=False),
                                         nn.AvgPool2d(kernel_size=2, stride=2)
                                         )

    def forward(self, RGBT):
        # 6
        RGB = RGBT[0]
        T = RGBT[1]
        x = torch.cat([RGB,T],1)
        x = self.base_feature(x)
        #print("RGB: ",RGB.shape)
        #print("x: ",x.shape)
        x1 = self.dense_layer1(torch.cat([x], 1))
        x2 = self.dense_layer2(torch.cat([x, x1], 1))
        x3 = self.dense_layer3(torch.cat([x, x1, x2], 1))
        x4 = self.dense_layer4(torch.cat([x, x1, x2, x3], 1))
        x5 = self.dense_layer5(torch.cat([x, x1, x2, x3, x4], 1))
        x6 = self.dense_layer6(torch.cat([x, x1, x2, x3, x4, x5], 1))
        x = self.transition1(torch.cat([x, x1, x2, x3, x4, x5, x6], 1))
        x = F.avg_pool2d(x, kernel_size=1, stride=1, padding=0)
        #print("x: ",x.shape)
        return x

def fusion_model():
    model = DenseNet()
    return model