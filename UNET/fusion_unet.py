import torch.nn as nn
import torch
from torch.nn import functional as F
import cv2


class FusionModel(nn.Module):
    def __init__(self, nin, nout):
        super(FusionModel, self).__init__()
		
        ratio=0.6
        
        c1 = int(64 * ratio)
        c2 = int(128 * ratio)
        c3 = int(256 * ratio)
        c4 = int(512 * ratio)

        self.block1 = Block([c1, c1,'M'], in_channels=3, first_block=True)
        self.block2 = Block([c2, c2,'M'], in_channels=c1)
        self.block3 = Block([c3, c3, c3, c3,'M'], in_channels=c2)
        self.block4 = Block([c4, c4, c4, c4, 'M'], in_channels=c3)
        self.block5 = Block([c4, c4, c4, c4], in_channels=c4)

        self.reg_layer = nn.Sequential(
            nn.Conv2d(c4, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )
        self._initialize_weights()
		#init unet
        self.in_conv = DoubleConv(nin, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // 2)
        self.up1 = Up(1024, 512 // 2)
        self.up2 = Up(512, 256 // 2)
        self.up3 = Up(256, 128 // 2)
        self.up4 = Up(128, 64)
        self.out_conv = OutConv(64, nout)
	

    def forward(self, RGBT):
        RGB = RGBT[0]
        T = RGBT[1]
        #print("RGB:",RGB.shape)
        rx1 = self.in_conv(RGB)
        #print("rx1:",rx1.shape)
        rx2 = self.down1(rx1)
        rx3 = self.down2(rx2)
        rx4 = self.down3(rx3)
        rx5 = self.down4(rx4)
        rx = self.up1(rx5,rx4)
        rx = self.up2(rx, rx3)
        rx = self.up3(rx, rx2)
        rx = self.up4(rx, rx1)
        rx = self.out_conv(rx)
        tx1 = self.in_conv(T)
        tx2 = self.down1(tx1)
        tx3 = self.down2(tx2)
        tx4 = self.down3(tx3)
        tx5 = self.down4(tx4)
        tx = self.up1(tx5, tx4)
        tx = self.up2(tx, tx3)
        tx = self.up3(tx, tx2)
        tx = self.up4(tx, tx1)
        tx = self.out_conv(tx)
        #print("tx:",tx.shape)
		
        RGB = rx
        T = tx
        RGB, T, shared = self.block1(RGB, T, None)
        RGB, T, shared = self.block2(RGB, T, shared)
        RGB, T, shared = self.block3(RGB, T, shared)
        RGB, T, shared = self.block4(RGB, T, shared)
        _, _, shared = self.block5(RGB, T, shared)
        x = shared
        #print("x: ",x.shape)
        #x = cv2.merge((RGB,T,shared))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        x = F.upsample_bilinear(x, scale_factor=2)
        x = self.reg_layer(x)
        #("x: ",x.shape)
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

class DoubleConv(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.double_conv = nn.Sequential(nn.Conv2d(nin, nout, 3, padding=1, stride=1),
                                         nn.BatchNorm2d(nout),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(nout, nout, 3, padding=1, stride=1),
                                         nn.BatchNorm2d(nout),
                                         nn.ReLU(inplace=True)
                                         )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.down_conv = nn.Sequential(nn.MaxPool2d(2),
                                       DoubleConv(nin, nout))

    def forward(self, x):
        return self.down_conv(x)


class Up(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.double_conv = DoubleConv(nin, nout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.double_conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, nin, nout):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


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
    #model = FusionModel()
    #return model
    #img = torch.rand([64, 1, 3, 3]).cuda()
    model = FusionModel(nin=3, nout=3).cuda()
    #print(model.forward(img).size())
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
