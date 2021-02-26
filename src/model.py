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
        channel_1 = 32
        channel_2 = 64
        channel_3 = 128
        channel_4 = 256
        channel_5 = 16
        self.encoder1 = ConvBlock(3, channel_1, kernel_size=3, stride=1, padding=1)
        self.encoder2 = ConvBlock(channel_1, channel_2, kernel_size=3, stride=1, padding=1)
        self.encoder3 = ConvBlock(channel_2, channel_3, kernel_size=3, stride=1, padding=1)
        self.encoder4 = ConvBlock(channel_3, channel_4, kernel_size=3, stride=1, padding=1)
        self.encoder5 = ConvBlock(channel_4, channel_5, kernel_size=3, stride=1, padding=1)
        self.decoder1 = DeConvBlock(channel_5, channel_4, kernel_size=3, stride=1, padding=1, dilation=1)
        self.decoder2 = DeConvBlock(channel_4, channel_3, kernel_size=3, stride=1, padding=1, dilation=1)
        self.decoder3 = DeConvBlock(channel_3, channel_2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.decoder4 = DeConvBlock(channel_2, channel_1, kernel_size=3, stride=1, padding=1, dilation=1)
        self.decoder5 = DeConvBlock(channel_1, 3, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv = nn.Conv2d(3, channel_1, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d((2, 2), stride = 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x): 
        feature_map = self.conv(x)
        feature_map = self.relu(feature_map)
        output = self.maxpool(feature_map)
        #print(output.shape)
        output = self.encoder2(output)
        #print(output.shape)
        output = self.encoder3(output)
        #print(output.shape)
        output = self.encoder4(output)
        #print(output.shape)
        output = self.encoder5(output)
        #print(feature_map.shape)
        #output = self.reparameterize(feature_map[:, :8, :, :], feature_map[:, 8:, :, :])
        output = self.decoder1(output)
        output = self.relu(output)
        output = self.decoder2(output)
        output = self.relu(output)
        output = self.decoder3(output)
        output = self.relu(output)
        output = self.decoder4(output)
        output = self.relu(output)
        output = self.decoder5(output)
        output = self.sigmoid(output)
        return output, feature_map

class Discriminator(nn.Module):
    def __init__(self):
        # Necessary
        super().__init__()
        channel_1 = 32
        channel_2 = 64
        channel_3 = 128
        channel_4 = 256
        channel_5 = 512
        self.encoder1 = ConvBlock(3, channel_1, kernel_size=3, stride=1, padding=1)
        self.encoder2 = ConvBlock(channel_1, channel_2, kernel_size=3, stride=1, padding=1)
        self.encoder3 = ConvBlock(channel_2, channel_3, kernel_size=3, stride=1, padding=1)
        self.encoder4 = ConvBlock(channel_3, channel_4, kernel_size=3, stride=1, padding=1)
        self.encoder5 = ConvBlock(channel_4, channel_5, kernel_size=3, stride=1, padding=1)
        self.decoder1 = DeConvBlock(channel_5, channel_4, kernel_size=3, stride=1, padding=1, dilation=1)
        self.decoder21 = DeConvBlock(channel_4*2, channel_3, kernel_size=3, stride=1, padding=1, dilation=1)
        self.decoder22 = nn.Conv2d(channel_3, channel_3, kernel_size=3, stride=1, padding=1, dilation=1)
        self.decoder31 = DeConvBlock(channel_3*2, channel_2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.decoder32 = nn.Conv2d(channel_2, channel_2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.decoder41 = DeConvBlock(channel_2*2, channel_1, kernel_size=3, stride=1, padding=1, dilation=1)
        self.decoder42 = nn.Conv2d(channel_1, channel_1, kernel_size=3, stride=1, padding=1, dilation=1)
        self.decoder5 = DeConvBlock(channel_1*2, 1, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv = nn.Conv2d(3, channel_1, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d((2, 2), stride = 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x): 
        feature_map = self.conv(x)
        feature_map = self.relu(feature_map)
        feature_map_1 = self.maxpool(feature_map)
        #print(output.shape)
        feature_map_2 = self.encoder2(feature_map_1)
        #print(output.shape)
        feature_map_3 = self.encoder3(feature_map_2)
        #print(output.shape)
        feature_map_4 = self.encoder4(feature_map_3)
        #print(output.shape)
        feature_map_5 = self.encoder5(feature_map_4)
        #print(feature_map.shape)
        #output = self.reparameterize(feature_map[:, :8, :, :], feature_map[:, 8:, :, :])
        output = self.decoder1(feature_map_5)
        output = self.relu(output)
        output = self.decoder21(torch.cat((feature_map_4, output), 1))
        output = self.relu(output)
        output = self.decoder22(output)
        output = self.relu(output)
        output = self.decoder31(torch.cat((feature_map_3, output), 1))
        output = self.relu(output)
        output = self.decoder32(output)
        output = self.relu(output)
        output = self.decoder41(torch.cat((feature_map_2, output), 1))
        output = self.decoder42(output)
        output = self.relu(output)
        output = self.decoder5(torch.cat((feature_map_1, output), 1))
        output = self.sigmoid(output)
        return output, feature_map
