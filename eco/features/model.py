# library
import os
from pathlib import Path
import json
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import models
from torchvision import transforms
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        # Necessary
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d((2, 2), stride = 2)
        #self.bn = nn.BatchNorm2d(out_channel, affine=False)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        #x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x
class DeConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, dilation):
        # Necessary
        super().__init__() 
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        #self.conv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        #self.bn = nn.BatchNorm2d(out_channel, affine=False)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
    def forward(self, x):
        x = self.up(x) 
        x = self.conv(x)
        #x = self.bn(x)
        return x

class FCNet(nn.Module):
    def __init__(self):
        # Necessary
        super().__init__()
        channel_1 = 16
        channel_2 = 64
        channel_3 = 128
        channel_4 = 128
        channel_5 = 32
        self.maxpool = nn.MaxPool2d((2, 2), stride = 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(3, channel_1, kernel_size=3, stride=1, padding=2)
        self.encoder2 = ConvBlock(channel_1, channel_2, kernel_size=3, stride=1, padding=1)
        self.encoder3 = ConvBlock(channel_2, channel_3, kernel_size=3, stride=1, padding=1)
        self.encoder4 = ConvBlock(channel_3, channel_4, kernel_size=3, stride=1, padding=1)
        # self.encoder5 = ConvBlock(channel_4, channel_5, kernel_size=3, stride=1, padding=1)
        # self.decoder1 = DeConvBlock(channel_5, channel_4, kernel_size=3, stride=1, padding=1, dilation=1)
        self.decoder2 = DeConvBlock(channel_4, channel_3, kernel_size=3, stride=1, padding=1, dilation=1)
        self.decoder3 = DeConvBlock(channel_3, channel_2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.decoder4 = DeConvBlock(channel_2, channel_1, kernel_size=3, stride=1, padding=1, dilation=1)
        self.decoder5 = DeConvBlock(channel_1, 3, kernel_size=3, stride=1, padding=2, dilation=1)
        
    def forward(self, x): 
        output = self.conv1(x)
        output = self.relu(output)
        feature_map_1 = self.maxpool(output)
        output = self.encoder2(feature_map_1)
        output = self.encoder3(output)
        feature_map_2 = self.encoder4(output)
        # output = self.encoder5(feature_map)
        # output = self.decoder1(output)
        # output = self.relu(output)
        output = self.decoder2(feature_map_2)
        output = self.relu(output)
        output = self.decoder3(output)
        output = self.relu(output)
        output = self.decoder4(output)
        output = self.relu(output)
        output = self.decoder5(output)
        output = self.sigmoid(output)
        return output, feature_map_1, feature_map_2

class Discriminator(nn.Module):
    def __init__(self):
        # Necessary
        super().__init__()
        channel_1 = 32
        channel_2 = 64
        channel_3 = 128
        channel_4 = 256
        self.maxpool = nn.MaxPool2d((2, 2), stride = 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # encoder
        self.conv1 = nn.Conv2d(3, channel_1, kernel_size=3, stride=1, padding=1)
        self.encoder2 = ConvBlock(channel_1, channel_2, kernel_size=3, stride=1, padding=1)
        self.encoder3 = ConvBlock(channel_2, channel_3, kernel_size=3, stride=1, padding=1)
        self.encoder4 = ConvBlock(channel_3, channel_4, kernel_size=3, stride=1, padding=1)
        # decoder
        self.decoder11 = DeConvBlock(channel_4, channel_3, kernel_size=3, stride=1, padding=1, dilation=1)
        self.decoder12 = nn.Conv2d(channel_3, channel_3, kernel_size=3, stride=1, padding=1, dilation=1)
        self.decoder21 = DeConvBlock(channel_3*2, channel_2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.decoder22 = nn.Conv2d(channel_2, channel_2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.decoder3 = DeConvBlock(channel_2*2, channel_1, kernel_size=3, stride=1, padding=1, dilation=1)
        self.decoder4 = DeConvBlock(channel_1, 1, kernel_size=3, stride=1, padding=1, dilation=1)
        # # segmentation
        # self.seg_decoder11 = DeConvBlock(channel_3, channel_2, kernel_size=3, stride=1, padding=1, dilation=1)
        # self.seg_decoder12 = nn.Conv2d(channel_2, channel_2, kernel_size=3, stride=1, padding=1, dilation=1)
        # self.seg_decoder21 = DeConvBlock(channel_2*2, channel_1, kernel_size=3, stride=1, padding=1, dilation=1)
        # self.seg_decoder22 = nn.Conv2d(channel_1, channel_1, kernel_size=3, stride=1, padding=1, dilation=1)
        # self.seg_decoder3 = DeConvBlock(channel_1*2, 1, kernel_size=5, stride=1, padding=2, dilation=1)
        
    def forward(self, x):
        # encoder
        feature_map = self.conv1(x)
        feature_map = self.relu(feature_map)
        feature_map_1 = self.maxpool(feature_map)
        feature_map_2 = self.encoder2(feature_map_1)
        feature_map_3 = self.encoder3(feature_map_2)
        feature_map_4 = self.encoder4(feature_map_3)
        # decoder
        output = self.decoder11(feature_map_4)
        output = self.relu(output)
        output = self.decoder12(output)
        output = self.relu(output)
        output = self.decoder21(torch.cat((feature_map_3, output), 1))
        output = self.relu(output)
        output = self.decoder22(output)
        output = self.relu(output)
        output = self.decoder3(torch.cat((feature_map_2, output), 1))
        output = self.relu(output)
        output = self.decoder4(output)
        output = self.sigmoid(output)
        # # segmentation
        # seg_output = self.seg_decoder11(feature_map_3)
        # seg_output = self.relu(seg_output)
        # seg_output = self.seg_decoder12(seg_output)
        # seg_output = self.relu(seg_output)
        # seg_output = self.seg_decoder21(torch.cat((feature_map_2, seg_output), 1))
        # seg_output = self.relu(seg_output)
        # seg_output = self.seg_decoder22(seg_output)
        # seg_output = self.relu(seg_output)
        # seg_output = self.seg_decoder3(torch.cat((feature_map_1, seg_output), 1))
        # seg_output = self.sigmoid(seg_output)
        return output, feature_map_1, feature_map_4

if __name__ == "__main__":
    a = FCNet().cuda()
    b = torch.zeros([1, 3, 735, 601])
    p, f = a(b.cuda())
    pr = torch.nn.functional.interpolate(p, (b.shape[2], b.shape[3]))
    print(pr.shape)
