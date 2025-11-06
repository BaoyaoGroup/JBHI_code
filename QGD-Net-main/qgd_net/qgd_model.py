import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from .qgd_parts import *


class QGDNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, num_groups=0):
        super(QGDNet, self).__init__()
        self.name = "QGDNet"
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64, num_groups=num_groups)
        self.down1 = Down(64, 128, num_groups=num_groups)
        self.down2 = Down(128, 256, num_groups=num_groups)
        self.down3 = Down(256, 512, num_groups=num_groups)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, num_groups=num_groups)
        self.up1 = Up(1024, 512 // factor, bilinear=bilinear, num_groups=num_groups)
        self.up2 = Up(512, 256 // factor, bilinear=bilinear, num_groups=num_groups)
        self.up3 = Up(256, 128 // factor, bilinear=bilinear, num_groups=num_groups)
        self.up4 = Up(128, 64, bilinear=bilinear, num_groups=num_groups)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        # print("x1", x1.shape)
        x2 = self.down1(x1)
        # print("x2", x2.shape)
        x3 = self.down2(x2)
        # print("x3", x3.shape)
        x4 = self.down3(x3)
        # print("x4", x4.shape)
        x5 = self.down4(x4)
        # print("x5", x5.shape)
        x = self.up1(x5, x4)
        # print("x", x.shape)
        x = self.up2(x, x3)
        # print("x", x.shape)
        x = self.up3(x, x2)
        # print("x", x.shape)
        x = self.up4(x, x1)
        # print("x", x.shape)
        logits = self.outc(x)
        # print("logits", logits.shape)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
