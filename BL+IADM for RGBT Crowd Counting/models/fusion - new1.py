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
            Conv2d(128+128, 128, 1, same_padding=True, NL='relu'),
            Conv2d(128, 256, 3, same_padding=True, NL='relu'),
            Conv2d(256, 256, 3, same_padding=True, NL='relu'),
            Conv2d(256, 256, 3, same_padding=True, NL='relu'),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.b4 = nn.Sequential(
            Conv2d(256, 512, 3, same_padding=True, NL='relu'),
            Conv2d(512, 512, 3, same_padding=True, NL='relu'),
            Conv2d(512, 512, 3, same_padding=True, NL='relu'),
        )

        self.block1 = Block([c1, c1, 'M'], in_channels=3, first_block=True)
        self.block2 = Block([c2, c2, 'M'], in_channels=c1)
        self.block3 = Block([c3, c3, c3, c3, 'M'], in_channels=c2)
        self.block4 = Block([c4, c4, c4, c4, 'M'], in_channels=c3)
        self.block5 = Block([c4, c4, c4, c4], in_channels=c4)

        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )
        self._initialize_weights()

    def forward(self, RGBT):
        RGB = RGBT[0]
        T = RGBT[1]
        RGB = self.b1(RGB)
        T = self.b1(T)
        RGB = self.b2(RGB)
        T = self.b2(T)
        x = torch.cat([RGB,T],1)
        x = self.b3(x)
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


def fusion_model():
    model = FusionModel()
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
