import torch
import torch.nn as nn
import torch.nn.functional as F

from ..initializer import create_initializer
from pytorch_wavelets import DWT1DForward, DWT1DInverse,DWTForward, DWTInverse # or simply DWT1D, IDWT1D


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,  # downsample with first conv
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))  # BN

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)  # apply ReLU after addition
        return y

class Block_lite_lite(nn.Module):
    def __init__(self, in_channels, out_channels,is_supplement =True,wavelet_name = "haar"):
        super().__init__()
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
        #Lnext = torch.cat([x,Lin],dim=1)
        ori = x
        x = self.bnx(x+ Lin)
        if is_downsample:
            Lin,Yh = self.dwt(Lin)
            ori,Yh = self.dwt(ori)
        Lnext = Lin + ori
        return x, Lnext
class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        bottleneck_channels = out_channels // self.expansion

        self.conv1 = nn.Conv2d(in_channels,
                               bottleneck_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,  # downsample with 3x3 conv
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        self.conv3 = nn.Conv2d(bottleneck_channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()  # identity
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))  # BN

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = F.relu(self.bn2(self.conv2(y)), inplace=True)
        y = self.bn3(self.conv3(y))  # not apply ReLU
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)  # apply ReLU after addition
        return y

class Network(nn.Module):
    def __init__(self, config,wavelet_name):
        super().__init__()

        model_config = config.model.resnet
        initial_channels = model_config.initial_channels
        block_type = model_config.block_type
        n_blocks = model_config.n_blocks

        assert block_type in ['basic', 'bottleneck']
        if block_type == 'basic':
            block = BasicBlock
        else:
            block = BottleneckBlock

        n_channels = [
            initial_channels,
            initial_channels * 2 * block.expansion,
            initial_channels * 4 * block.expansion,
            initial_channels * 8 * block.expansion,
        ]

        self.conv = nn.Conv2d(config.dataset.n_channels,
                              n_channels[0],
                              kernel_size=7,
                              stride=2,
                              padding=3,
                              bias=False)
        self.bn = nn.BatchNorm2d(initial_channels)

        self.stage1 = self._make_stage(n_channels[0],
                                       n_channels[0],
                                       n_blocks[0],
                                       block,
                                       stride=1)
        self.stage2 = self._make_stage(n_channels[0],
                                       n_channels[1],
                                       n_blocks[1],
                                       block,
                                       stride=2)
        self.stage3 = self._make_stage(n_channels[1],
                                       n_channels[2],
                                       n_blocks[2],
                                       block,
                                       stride=2)
        self.stage4 = self._make_stage(n_channels[2],
                                       n_channels[3],
                                       n_blocks[3],
                                       block,
                                       stride=2)

        self.dwtLx128 = DWTForward(J=1, mode='periodization', wave=wavelet_name)
        self.dwtLx64 = DWTForward(J=2, mode='periodization', wave=wavelet_name)
        self.dwtLx32 = DWTForward(J=3, mode='periodization', wave=wavelet_name)
        self.dwtLx16 = DWTForward(J=4, mode='periodization', wave=wavelet_name)
        self.dwtLx8 = DWTForward(J=5, mode='periodization', wave=wavelet_name)
        
        self.pool1 = nn.MaxPool2d(3,stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(3,stride=2, padding=1)
        self.pool3 = nn.MaxPool2d(3,stride=2, padding=1)
        self.pool4 = nn.MaxPool2d(3,stride=2, padding=1)
        self.pool5 = nn.MaxPool2d(3,stride=2, padding=1)
        
        print(wavelet_name)
        
        ##L1
        self.wavelet_conv = Block_lite_lite(in_channels = 3, out_channels = 64, is_supplement = True, wavelet_name = wavelet_name)
        self.wavelet_pool = Block_lite_lite(in_channels = 64, out_channels = 64, is_supplement = True, wavelet_name = wavelet_name)
        self.wavelet_1 = Block_lite_lite(in_channels = 64, out_channels = 64, is_supplement = True, wavelet_name = wavelet_name)
        self.wavelet_2 = Block_lite_lite(in_channels = 64, out_channels = 128, is_supplement = True, wavelet_name = wavelet_name)
        self.wavelet_3 = Block_lite_lite(in_channels = 128, out_channels = 256, is_supplement = True, wavelet_name = wavelet_name)
        self.wavelet_4 = Block_lite_lite(in_channels = 256, out_channels = 512, is_supplement = True, wavelet_name = wavelet_name)

        # compute conv feature size
        with torch.no_grad():
            dummy_data = torch.zeros(
                (1, config.dataset.n_channels, config.dataset.image_size,
                 config.dataset.image_size),
                dtype=torch.float32)
            self.feature_size = self._forward_conv(dummy_data).view(
                -1).shape[0]

        self.fc = nn.Linear(self.feature_size, config.dataset.n_classes)

        # initialize weights
        initializer = create_initializer(config.model.init_mode)
        self.apply(initializer)

    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = f'block{index + 1}'
            if index == 0:
                stage.add_module(
                    block_name, block(in_channels, out_channels,
                                      stride=stride))
            else:
                stage.add_module(block_name,
                                 block(out_channels, out_channels, stride=1))
        return stage

    def _forward_conv(self, x):
        
        Lx128,Yh = self.dwtLx128(x)
        Lx64,Yh = self.dwtLx64(x)
        Lx32,Yh = self.dwtLx32(x)
        Lx16,Yh = self.dwtLx16(x)
        Lx8,Yh = self.dwtLx8(x)
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x , Lnext = self.wavelet_conv(Llast = Lx128, original = Lx128, x = x, is_downsample = True)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x , Lnext = self.wavelet_pool(Llast = Lnext, original = Lx64, x = x, is_downsample = False)
        x = self.stage1(x)
        x , Lnext = self.wavelet_1(Llast = Lnext, original = Lx64, x = x, is_downsample = True)
        x = self.stage2(x)
        x , Lnext = self.wavelet_2(Llast = Lnext, original = Lx32, x = x, is_downsample = True)
        x = self.stage3(x)
        x , Lnext = self.wavelet_3(Llast = Lnext, original = Lx16, x = x, is_downsample = True)
        x = self.stage4(x)
        x , Lnext = self.wavelet_4(Llast = Lnext, original = Lx8, x = x, is_downsample = False)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x