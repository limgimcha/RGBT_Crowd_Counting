import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.autograd import Variable

import cv2
import PIL
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np

class FusionModel(nn.Module):
    def __init__(self, block, num_blocks):
        super(FusionModel, self).__init__()

        ratio=0.6

        c1 = int(64 * ratio)
        c2 = int(128 * ratio)
        c3 = int(256 * ratio)
        c4 = int(512 * ratio)

        self.block1 = Block([c2, c2,'M'], in_channels=c2, first_block=True)
        self.block2 = Block([c2, c2,'M'], in_channels=c2)
        self.block3 = Block([c2, c2, c2, c2,'M'], in_channels=c2)
        self.block4 = Block([c3, c3, c3, c3,'M'], in_channels=c2)
        self.block5 = Block([c4, c4, c4, c4], in_channels=c3)

        self.reg_layer = nn.Sequential(
            nn.Conv2d(c4, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )
        self._initialize_weights()
        
        #fpn init
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(256, c2, kernel_size=1, stride=1, padding=0, bias=False)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)


    def forward(self, RGBT):
        RGB = RGBT[0]
        T = RGBT[1]

        # Bottom-up
        rc1 = F.relu(self.bn1(self.conv1(RGB)))
        #rc1 = F.max_pool2d(rc1, kernel_size=3, stride=2, padding=1)
        rc2 = self.layer1(rc1)
        rc3 = self.layer2(rc2)
        rc4 = self.layer3(rc3)
        rc5 = self.layer4(rc4)
        # Top-down
        rp5 = self.toplayer(rc5)
        rp4 = self._upsample_add(rp5, self.latlayer1(rc4))
        rp3 = self._upsample_add(rp4, self.latlayer2(rc3))
        rp2 = self._upsample_add(rp3, self.latlayer3(rc2))
        # Smooth
        rp4 = self.smooth1(rp4)
        rp3 = self.smooth2(rp3)
        rp2 = self.smooth3(rp2)

        # Bottom-up
        #print("T: ",T.shape)
        tc1 = F.relu(self.bn1(self.conv1(T)))
        #print("tc1: ",tc1.shape)
        #tc1 = F.max_pool2d(tc1, kernel_size=3, stride=2, padding=1)
        #print("tc1: ",tc1.shape)
        tc2 = self.layer1(tc1)
        #print("tc2: ",tc2.shape)
        tc3 = self.layer2(tc2)
        tc4 = self.layer3(tc3)
        tc5 = self.layer4(tc4)
        # Top-down
        tp5 = self.toplayer(tc5)
        tp4 = self._upsample_add(tp5, self.latlayer1(tc4))
        tp3 = self._upsample_add(tp4, self.latlayer2(tc3))
        tp2 = self._upsample_add(tp3, self.latlayer3(tc2))
        # Smooth
        tp4 = self.smooth1(tp4)
        tp3 = self.smooth2(tp3)
        tp2 = self.smooth3(tp2)
        #print("tp2: ",tp2.shape)
        
        '''
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 3, 1) # rows, cols, index
        output1 = rp2[0][0].cpu().detach().numpy()
        ax1.set_title("p2 Image")
        ax1.imshow(output1)

        ax2 = fig.add_subplot(2, 3, 2) # rows, cols, index
        output1 = rp3[0][0].cpu().detach().numpy()
        ax2.set_title("p3 Image")
        ax2.imshow(output1)

        ax3 = fig.add_subplot(2, 3, 3) # rows, cols, index
        output1 = rp4[0][0].cpu().detach().numpy()
        ax3.set_title("p4 Image")
        ax3.imshow(output1)

        ax4 = fig.add_subplot(2, 3, 4) # rows, cols, index
        output1 = RGB[0].cpu().permute(1,2,0).detach().numpy()
        ax4.set_title("rgb Image")
        ax4.imshow(output1)
        '''
        
        

        #plt.imshow(RGB[0].cpu().permute(1,2,0).detach().numpy())
        #plt.show()
        rp2 = self.conv2(rp2)
        tp2 = self.conv2(tp2)
        RGB = rp2
        T = tp2

        RGB, T, shared = self.block1(RGB, T, None)
        #print("shared: ",shared.shape)
        #RGB, T, shared = self.block2(RGB, T, shared)
        #print("shared: ",shared.shape)
        RGB, T, shared = self.block3(RGB, T, shared)
        #print("shared: ",shared.shape)
        RGB, T, shared = self.block4(RGB, T, shared)
        #print("shared: ",shared.shape)
        _, _, shared = self.block5(RGB, T, shared)
        #print("shared: ",shared.shape)
        x = shared

        #print("x: ",x.shape)
        #x = F.upsample_bilinear(x, scale_factor=2)
        #print("x: ",x.shape)
        x = self.reg_layer(x)
        #print("x: ",x.shape)
        
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

    #fpn method
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y
    

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Block(nn.Module):
    def __init__(self, cfg, in_channels, first_block=False, dilation_rate=1):
        super(Block, self).__init__()
        self.seen = 0
        self.first_block = first_block
        self.d_rate = dilation_rate

        self.rgb_conv = make_layers(cfg, in_channels=in_channels,batch_norm=True, d_rate=self.d_rate)
        self.t_conv = make_layers(cfg, in_channels=in_channels,batch_norm=True, d_rate=self.d_rate)
        if first_block is False:
            self.shared_conv = make_layers(cfg, in_channels=in_channels,batch_norm=True, d_rate=self.d_rate)

        channels = cfg[0]
        self.rgb_msc = MSC(channels)
        self.t_msc = MSC(channels)
        if first_block is False:
            self.shared_fuse_msc = MSC(channels)
        self.shared_distribute_msc = MSC(channels)

        self.rgb_fuse_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.t_fuse_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.rgb_distribute_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.t_distribute_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, RGB, T, shared):
        RGB = self.rgb_conv(RGB)
        T = self.t_conv(T)
        if self.first_block:
            shared = torch.zeros(RGB.shape).cuda()
        else:
            shared = self.shared_conv(shared)

        new_RGB, new_T, new_shared = self.fuse(RGB, T, shared)
        return new_RGB, new_T, new_shared

    def fuse(self, RGB, T, shared):

        RGB_m = self.rgb_msc(RGB)
        T_m = self.t_msc(T)
        if self.first_block:
            shared_m = shared  # zero
        else:
            shared_m = self.shared_fuse_msc(shared)

        rgb_s = self.rgb_fuse_1x1conv(RGB_m - shared_m)
        rgb_fuse_gate = torch.sigmoid(rgb_s)
        t_s = self.t_fuse_1x1conv(T_m - shared_m)
        t_fuse_gate = torch.sigmoid(t_s)
        new_shared = shared + (RGB_m - shared_m) * rgb_fuse_gate + (T_m - shared_m) * t_fuse_gate

        new_shared_m = self.shared_distribute_msc(new_shared)
        s_rgb = self.rgb_distribute_1x1conv(new_shared_m - RGB_m)
        rgb_distribute_gate = torch.sigmoid(s_rgb)
        s_t = self.t_distribute_1x1conv(new_shared_m - T_m)
        t_distribute_gate = torch.sigmoid(s_t)
        new_RGB = RGB + (new_shared_m - RGB_m) * rgb_distribute_gate
        new_T = T + (new_shared_m - T_m) * t_distribute_gate

        return new_RGB, new_T, new_shared


class MSC(nn.Module):
    def __init__(self, channels):
        super(MSC, self).__init__()
        self.channels = channels
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.conv = nn.Sequential(
            nn.Conv2d(3*channels, channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = nn.functional.interpolate(self.pool1(x), x.shape[2:])
        x2 = nn.functional.interpolate(self.pool2(x), x.shape[2:])
        concat = torch.cat([x, x1, x2], 1)
        fusion = self.conv(concat)
        return fusion


def fusion_model():
    #model = FusionModel()
    model = FusionModel(Bottleneck, [2,2,2,2])
    return model


def make_layers(cfg, in_channels=3, batch_norm=False, d_rate=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

