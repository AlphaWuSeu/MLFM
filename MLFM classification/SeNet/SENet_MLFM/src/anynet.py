"""
@author: Signatrix GmbH
Implementation of paradigm described in paper: Designing Network Design Spaces published by Facebook AI Research (FAIR)
"""
import torch
import torch.nn as nn
from src.modules import Stem, Stage, Head
from src.config import NUM_CLASSES
import torch.nn.functional as F
import math
from math import sqrt

from pytorch_wavelets import DWT1DForward, DWT1DInverse,DWTForward, DWTInverse # or simply DWT1D, IDWT1D


class Block_lite(nn.Module):
    def __init__(self, in_channels, out_channels,is_supplement =True,wavelet_name = "haar"):
        super().__init__()
        #input gate
        self.convinput = nn.Conv2d(in_channels,out_channels,kernel_size=1)
        self.bninput = nn.BatchNorm2d(out_channels)
        
        self.is_supplement = is_supplement
        if self.is_supplement:
            self.convsupplement = nn.Conv2d(3,out_channels,kernel_size=3, stride=1, padding=1)
            self.bnsupplement = nn.BatchNorm2d(out_channels)
        
        
        self.bnx = nn.BatchNorm2d(out_channels)
        self.dwt = DWTForward(J=1, mode='periodization', wave=wavelet_name)
        
        
    def forward(self, Llast,original,x,is_downsample = True):
        
        Lin = F.relu(self.bninput(self.convinput(Llast)), inplace=True)
        if self.is_supplement:
            Lsupp = F.relu(self.bnsupplement(self.convsupplement(original)), inplace=True)
            Lin = Lin + Lsupp
        # Lnext = torch.cat([x,Lin],dim=1)
        Lnext = x + Lin
        x = self.bnx(x+ Lin)
        if is_downsample:
            Lnext,Yh = self.dwt(Lnext)
        return x, Lnext
class AnyNetX(nn.Module):
# class AnyNetX_wavelet(nn.Module):
    def __init__(self, ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio):
        super(AnyNetX, self).__init__()
        # For each stage, at each layer, number of channels (block width / bottleneck ratio) must be divisible by group width
        for block_width, bottleneck_ratio, group_width in zip(ls_block_width, ls_bottleneck_ratio, ls_group_width):
            assert block_width % (bottleneck_ratio * group_width) == 0
        self.net = nn.Sequential()
        prev_block_width = 32
        self.net.add_module("stem", Stem(prev_block_width))

        for i, (num_blocks, block_width, bottleneck_ratio, group_width) in enumerate(zip(ls_num_blocks, ls_block_width,
                                                                                         ls_bottleneck_ratio,
                                                                                         ls_group_width)):
            self.net.add_module("stage_{}".format(i),
                                Stage(num_blocks, prev_block_width, block_width, bottleneck_ratio, group_width, stride, se_ratio))
            prev_block_width = block_width
        self.net.add_module("head", Head(ls_block_width[-1], NUM_CLASSES))
        wavelet_name = "haar"
        self.dwtLx128 = DWTForward(J=1, mode='periodization', wave=wavelet_name)
        self.dwtLx64 = DWTForward(J=2, mode='periodization', wave=wavelet_name)
        self.dwtLx32 = DWTForward(J=3, mode='periodization', wave=wavelet_name)
        self.dwtLx16 = DWTForward(J=4, mode='periodization', wave=wavelet_name)
        self.dwtLx8 = DWTForward(J=5, mode='periodization', wave=wavelet_name)
        self.wavelet0 = Block_lite(in_channels = 3, out_channels = 32, is_supplement = True, wavelet_name = "haar")
        self.wavelet1 = Block_lite(in_channels=32, out_channels=48, is_supplement=True, wavelet_name="haar")
        self.wavelet2 = Block_lite(in_channels=48, out_channels=104, is_supplement=True, wavelet_name="haar")
        self.wavelet3 = Block_lite(in_channels=104, out_channels=208, is_supplement=True, wavelet_name="haar")
        self.wavelet4 = Block_lite(in_channels=208, out_channels=448, is_supplement=True, wavelet_name="haar")

        self.initialize_weight()

    def initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.01)
                m.bias.data.zero_()

    def forward(self, x):
        # x = self.net(x)
        # for feature in self.net:
        #     x = feature(x)
        # print("33333",x.shape)
        Lx128,Yh = self.dwtLx128(x)
        Lx64,Yh = self.dwtLx64(x)
        Lx32,Yh = self.dwtLx32(x)
        Lx16,Yh = self.dwtLx16(x)
        Lx8,Yh = self.dwtLx8(x)
        
        feature0 = self.net[0]
        feature1 = self.net[1]
        feature2 = self.net[2]
        feature3 = self.net[3]
        feature4 = self.net[4]
        feature5 = self.net[5]
        x = feature0(x)
        # print(x.shape)#[1, 32, 128, 128]
        x , Lnext = self.wavelet0(Llast = Lx128, original = Lx128, x = x, is_downsample = True)
        x = feature1(x)
        # print(x.shape)#[1, 48, 64, 64]
        x, Lnext = self.wavelet1(Llast=Lnext, original=Lx64, x = x, is_downsample=True)
        x = feature2(x)
        # print(x.shape)#[1, 104, 32, 32]
        x, Lnext = self.wavelet2(Llast=Lnext, original=Lx32, x=x, is_downsample=True)
        x = feature3(x)
        # print(x.shape)#[1, 208, 16, 16]
        x, Lnext = self.wavelet3(Llast=Lnext, original=Lx16, x=x, is_downsample=True)
        x = feature4(x)
        # print(x.shape)#[1, 448, 8, 8]
        x, Lnext = self.wavelet4(Llast=Lnext, original=Lx8, x=x, is_downsample=False)
        x = feature5(x)
        # print(x.shape)#[1, 100]
        # sys.exit()
        return x

#class AnyNetX(nn.Module):
class AnyNetX_original(nn.Module):

    def __init__(self, ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio):
        super(AnyNetX, self).__init__()
        # For each stage, at each layer, number of channels (block width / bottleneck ratio) must be divisible by group width
        for block_width, bottleneck_ratio, group_width in zip(ls_block_width, ls_bottleneck_ratio, ls_group_width):
            assert block_width % (bottleneck_ratio * group_width) == 0
        self.net = nn.Sequential()
        prev_block_width = 32
        self.net.add_module("stem", Stem(prev_block_width))

        for i, (num_blocks, block_width, bottleneck_ratio, group_width) in enumerate(zip(ls_num_blocks, ls_block_width,
                                                                                         ls_bottleneck_ratio,
                                                                                         ls_group_width)):
            self.net.add_module("stage_{}".format(i),
                                Stage(num_blocks, prev_block_width, block_width, bottleneck_ratio, group_width, stride, se_ratio))
            prev_block_width = block_width
        self.net.add_module("head", Head(ls_block_width[-1], NUM_CLASSES))

        self.initialize_weight()

    def initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.net(x)
        return x
class AnyNetXb(AnyNetX):
    def __init__(self, ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio):
        super(AnyNetXb, self).__init__(ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio)
        assert len(set(ls_bottleneck_ratio)) == 1


class AnyNetXc(AnyNetXb):
    def __init__(self, ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio):
        super(AnyNetXc, self).__init__(ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio)
        assert len(set(ls_group_width)) == 1


class AnyNetXd(AnyNetXc):
    def __init__(self, ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio):
        super(AnyNetXd, self).__init__(ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio)
        assert all(i <= j for i, j in zip(ls_block_width, ls_block_width[1:])) is True


class AnyNetXe(AnyNetXd):
    def __init__(self, ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio):
        super(AnyNetXe, self).__init__(ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio)
        if len(ls_num_blocks > 2):
            assert all(i <= j for i, j in zip(ls_num_blocks[:-2], ls_num_blocks[1:-1])) is True
