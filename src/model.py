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
class Net(nn.Module):
    def __init__(self):
        # Necessary
        super().__init__()
        channel_1 = 32
        channel_2 = 64
        channel_3 = 4
        self.encoder1 = ConvBlock(3, channel_1, kernel_size=3, stride=1, padding=1)
        self.encoder2 = ConvBlock(channel_1, channel_2, kernel_size=3, stride=1, padding=1)
        self.encoder3 = ConvBlock(channel_2, channel_3, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Flatten()
        self.decoder1 = DeConvBlock(channel_3, channel_2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.decoder2 = DeConvBlock(channel_2, channel_1, kernel_size=3, stride=1, padding=1, dilation=1)
        self.decoder3 = DeConvBlock(channel_1, 3, kernel_size=3, stride=1, padding=1, dilation=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid() 
    def forward(self, x): 
        output = self.encoder1(x)
        output = self.encoder2(output)
        output = self.encoder3(output)
        temp_1 = output.shape[1]
        temp_2 = output.shape[2]
        temp_3 = output.shape[3]
        output = self.fc1(output)
        output_fc1 = output
        output = output.reshape(-1, temp_1, temp_2, temp_3)
        output = self.decoder1(output)
        output = self.relu(output)
        output = self.decoder2(output)
        output = self.relu(output)
        output = self.decoder3(output)
        output = self.sigmoid(output)
        return output, output_fc1
class VAENet(nn.Module):
    def __init__(self):
        # necessary
        super().__init__()
        channel_1 = 32
        channel_2 = 64
        channel_3 = 4
        self.encoder1 = ConvBlock(3, channel_1, kernel_size=3, stride=1, padding=1)
        self.encoder2 = ConvBlock(channel_1, channel_2, kernel_size=3, stride=1, padding=1)
        self.encoder3 = ConvBlock(channel_2, channel_3, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Flatten()
        # VAE
        self.fc22 = nn.Linear(1024, 32)
        self.fc23 = nn.Linear(1024, 32)
        self.fc3 = nn.Linear(32, 1024)
        self.decoder1 = DeConvBlock(channel_3, channel_2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.decoder2 = nn.Conv2d(channel_2, channel_2, kernel_size=3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(channel_2, channel_2, kernel_size=3, stride=1, padding=1)
        self.decoder4 = DeConvBlock(channel_2, channel_1, kernel_size=3, stride=1, padding=1, dilation=1)
        self.decoder5 = DeConvBlock(channel_1, 3, kernel_size=3, stride=1, padding=1, dilation=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid() 
        self.bn = nn.BatchNorm2d(channel_2)
    # VAE
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    def forward(self, x): 
        output = self.encoder1(x)
        output = self.encoder2(output)
        output = self.encoder3(output)
        temp_1 = output.shape[1]
        temp_2 = output.shape[2]
        temp_3 = output.shape[3]
        output = self.fc1(output)
        output_fc1 = output
        # mean, var
        mu = self.fc22(output)
        logvar = self.fc23(output)
        output = self.reparameterize(mu, logvar)
        output = self.fc3(output)
        output = output.reshape(-1, temp_1, temp_2, temp_3)
        output = self.decoder1(output)
        output = self.relu(output)
        output = self.decoder2(output)
        output = self.bn(output)
        output = self.relu(output)
        output = self.decoder3(output)
        output = self.bn(output)
        output = self.relu(output)
        output = self.decoder4(output)
        output = self.relu(output)
        output = self.decoder5(output)
        output = self.sigmoid(output)
        # inference
        inference = self.fc3(mu)
        inference = inference.reshape(-1, temp_1, temp_2, temp_3)
        inference = self.decoder1(inference)
        inference = self.relu(inference)
        inference = self.decoder2(inference)
        inference = self.bn(inference)
        inference = self.relu(inference)
        inference = self.decoder3(inference)
        inference = self.bn(inference)
        inference = self.relu(inference)
        inference = self.decoder4(inference)
        inference = self.relu(inference)
        inference = self.decoder5(inference)
        inference = self.sigmoid(inference)

        return output, mu, logvar, inference, output_fc1

class FCNet(nn.Module):
    def __init__(self):
        # Necessary
        super().__init__()
        channel_1 = 32
        channel_2 = 64
        channel_3 = 128
        channel_4 = 256
        channel_5 = 1024
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
        self.maxpool = nn.MaxPool2d((2, 2), stride = 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x): 
        feature_map = self.encoder1(x)
        #print(output.shape)
        output = self.encoder2(feature_map)
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
