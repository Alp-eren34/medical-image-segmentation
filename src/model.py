import torch
import torch.nn as nn
import torchvision.transforms.functional as tf

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=2):
        super(UNet, self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = DoubleConv(in_channels, 64)
        self.down_conv_2 = DoubleConv(64, 128)
        self.down_conv_3 = DoubleConv(128, 256)
        self.down_conv_4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.up_conv_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.double_conv_up_1 = DoubleConv(1024, 512)
        self.double_conv_up_2 = DoubleConv(512, 256)
        self.double_conv_up_3 = DoubleConv(256, 128)
        self.double_conv_up_4 = DoubleConv(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        x = self.down_conv_1(x)
        skip_connections.append(x)
        x = self.max_pool_2x2(x)
        x = self.down_conv_2(x)
        skip_connections.append(x)
        x = self.max_pool_2x2(x)
        x = self.down_conv_3(x)
        skip_connections.append(x)
        x = self.max_pool_2x2(x)
        x = self.down_conv_4(x)
        skip_connections.append(x)
        x = self.max_pool_2x2(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        x = self.up_conv_1(x)
        x = torch.cat((skip_connections[0], x), dim=1)
        x = self.double_conv_up_1(x)
        x = self.up_conv_2(x)
        x = torch.cat((skip_connections[1], x), dim=1)
        x = self.double_conv_up_2(x)
        x = self.up_conv_3(x)   
        x = torch.cat((skip_connections[2], x), dim=1)
        x = self.double_conv_up_3(x)
        x = self.up_conv_4(x)
        x = torch.cat((skip_connections[3], x), dim=1)
        x = self.double_conv_up_4(x)
        x = self.final_conv(x)
        return x