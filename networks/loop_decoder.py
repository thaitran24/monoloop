# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *


class LoopDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=3, use_skips=True):
        super(LoopDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features, disp_maps=None):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                if disp_maps:
                    disp = disp_maps[("disp", i)].expand(-1, disp_maps[("disp", i)].shape[1], -1, -1)
                    x = torch.mul(x, disp)
                self.outputs[("render", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
        return self.outputs


class CycleGANDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=3, use_skips=True, use_bias=True, device='cuda'):
        super(CycleGANDecoder, self).__init__()
        self.num_ch_enc = num_ch_enc
        self.num_ch_enc = np.array([64, 128, 256, 512, 512])
        self.num_ch_dec = np.array([64, 64, 128, 256, 512])
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.use_bias = use_bias
        self.scales = scales

        self.convs = OrderedDict()
        self.relu = nn.ReLU(True)
        self.reflection = nn.ReflectionPad2d(3)
        self.conv = nn.Conv2d(64, 3, kernel_size=7, padding=0)

        for i in range(4, -1, -1):
            num_ch_in = self.num_ch_enc[i]
            if i > 0:
                num_ch_out = self.num_ch_enc[i - 1]
            else:
                num_ch_in = self.num_ch_enc[i + 1]
                num_ch_out = self.num_ch_enc[i]

            self.convs[("upconv", i, 0)] = nn.ConvTranspose2d(num_ch_in, num_ch_out,
                                         kernel_size=3, stride=2, padding=1, 
                                         output_padding=1, bias=use_bias).to(device)
            self.convs[("norm", i)] = nn.InstanceNorm2d(num_ch_out).to(device)

            num_ch_in = self.num_ch_dec[i]
            if i > 0:
                num_ch_out = self.num_ch_dec[i - 1]
            else:
                num_ch_in = self.num_ch_dec[i + 1]
                num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = nn.Conv2d(num_ch_in, num_ch_out,
                                         kernel_size=3, padding=1, bias=use_bias).to(device)

        self.decoder = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features, disp_maps=None):
        self.outputs = {}
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [self.convs[("upconv", i, 1)](x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            if i in self.scales:
                if disp_maps:
                    disp = disp_maps[("disp", i)].expand(-1, disp_maps[("disp", i)].shape[1], -1, -1)
                    x = torch.mul(x, disp)
                x = self.convs[("norm", i)](x)
                x = self.relu(x)
                if i == 0:
                    x = nn.Tanh()(x)
                    x = self.conv(x)
                    x = self.reflection(x)
                self.outputs[("render", i)] = x

        return self.outputs