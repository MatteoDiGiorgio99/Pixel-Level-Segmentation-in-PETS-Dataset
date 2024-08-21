
import torch
from torch import nn
from Config import DROPOUT_PROB
import torch.nn.functional as F

#Define the SegNet architecture

class SegNet(nn.Module):
    def __init__(self, n_classes=3):
        super(SegNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 512)

        # Decoder
        self.dec5 = self.conv_block(512, 512)
        self.dec4 = self.conv_block(512, 512)
        self.dec3 = self.conv_block(512, 256)
        self.dec2 = self.conv_block(256, 128)
        self.dec1 = self.conv_block(128, 64)

        self.output = nn.Conv2d(64, n_classes, kernel_size=1)
        self.output_activation = nn.Softmax(dim=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        x1, indices1 = F.max_pool2d(self.enc1(x), kernel_size=2, stride=2, return_indices=True)
        x2, indices2 = F.max_pool2d(self.enc2(x1), kernel_size=2, stride=2, return_indices=True)
        x3, indices3 = F.max_pool2d(self.enc3(x2), kernel_size=2, stride=2, return_indices=True)
        x4, indices4 = F.max_pool2d(self.enc4(x3), kernel_size=2, stride=2, return_indices=True)
        x5, indices5 = F.max_pool2d(self.enc5(x4), kernel_size=2, stride=2, return_indices=True)

        # Decoder
        d5 = F.max_unpool2d(self.dec5(x5), indices5, kernel_size=2, stride=2)
        d4 = F.max_unpool2d(self.dec4(d5), indices4, kernel_size=2, stride=2)
        d3 = F.max_unpool2d(self.dec3(d4), indices3, kernel_size=2, stride=2)
        d2 = F.max_unpool2d(self.dec2(d3), indices2, kernel_size=2, stride=2)
        d1 = F.max_unpool2d(self.dec1(d2), indices1, kernel_size=2, stride=2)

        out = self.output(d1)
        out = self.output_activation(out)
        return out

