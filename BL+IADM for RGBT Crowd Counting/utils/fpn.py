'''FPN in PyTorch.

See the paper "Feature Pyramid Networks for Object Detection" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fusion import *
from torch.autograd import Variable

class Block(nn.Module):
    def __init__(self, cfg, in_channels, first_block=False, dilation_rate=1):
        super(Block, self).__init__()
        self.seen = 0
        self.first_block = first_block
        self.d_rate = dilation_rate

        self.rgb_conv = make_layers(cfg, in_channels=in_channels, d_rate=self.d_rate)
        self.t_conv = make_layers(cfg, in_channels=in_channels, d_rate=self.d_rate)
        if first_block is False:
            self.shared_conv = make_layers(cfg, in_channels=in_channels, d_rate=self.d_rate)

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


class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64
        ratio = 0.6

        c1 = int(64 * ratio)
        c2 = int(128 * ratio)
        c3 = int(256 * ratio)
        c4 = int(512 * ratio)

        self.block1 = Block([c1, c1, 'M'], in_channels=3, first_block=True)
        self.block2 = Block([c2, c2, 'M'], in_channels=c1)
        self.block3 = Block([c3, c3, c3, c3, 'M'], in_channels=c2)
        self.block4 = Block([c4, c4, c4, c4, 'M'], in_channels=c3)
        self.block5 = Block([c4, c4, c4, c4], in_channels=c4)


        self.conv1 = nn.Conv2d(3, int(ratio*64), kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

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

    def forward(self, x):
        print("타입은???", type(x))
        RGB = x[0]
        T = x[1]

        RGB, T, shared = self.block1(RGB, T, None)
        RGB, T, shared = self.block2(RGB, T, shared)
        RGB, T, shared = self.block3(RGB, T, shared)
        RGB, T, shared = self.block4(RGB, T, shared)
        _, _, shared = self.block5(RGB, T, shared)
        x = shared
        x = F.upsample_bilinear(x, scale_factor=2)

        # print("x is*****", x)

        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        return torch.abs(p2)


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


def FPN101():
    # return FPN(Bottleneck, [2,4,23,3])
    return FPN(Bottleneck, [2,2,2,2])