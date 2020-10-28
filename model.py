# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)

class Hide(nn.Module):
    def __init__(self):
        super(Hide, self).__init__()
        self.prepare = nn.Sequential(
            conv3x3(3, 64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(True)
        )
        self.hidding_1 = nn.Sequential(
            nn.Conv2d(128, 64, 1, 1, 0),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            conv3x3(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            conv3x3(64, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        self.hidding_2 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            conv3x3(32, 16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            conv3x3(16, 3),
            nn.Tanh()
        )
    def forward(self, secret, cover):
        sec_feature = self.prepare(secret)
        cover_feature = self.prepare(cover)
        out = self.hidding_1(torch.cat([sec_feature, cover_feature], dim=1))
        out = self.hidding_2(out)
        return out

class Reveal(nn.Module):
    def __init__(self):
        super(Reveal, self).__init__()
        self.reveal = nn.Sequential(
            conv3x3(3, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            conv3x3(32, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            conv3x3(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            conv3x3(64, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            conv3x3(32, 16),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            conv3x3(16, 3),
            nn.Tanh()
        )

    def forward(self, image):
        out = self.reveal(image)
        return out

