
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


from pytorch_wavelets import DWT1DForward, DWT1DInverse,DWTForward, DWTInverse # or simply DWT1D, IDWT1D

__all__ = ['mobilenetv2']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)
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
        Lnext = torch.cat([x,Lin],dim=1)
        x = self.bnx(x+ Lin)
        if is_downsample:
            Lnext,Yh = self.dwt(Lnext)
        return x, Lnext
class Block_lite_lite(nn.Module):
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
        #Lnext = torch.cat([x,Lin],dim=1)
        ori = x
        x = self.bnx(x+ Lin)
        if is_downsample:
            Lin,Yh = self.dwt(Lin)
            ori,Yh = self.dwt(ori)
        Lnext = Lin + ori
        return x, Lnext
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
class MobileNetV2_wavelet(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.):
        super(MobileNetV2_wavelet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        wavelet_name = "haar"
        # building first layer
        
        # building inverted residual blocks
        
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280

        
        self.dwtLx128 = DWTForward(J=1, mode='periodization', wave=wavelet_name)
        self.dwtLx64 = DWTForward(J=2, mode='periodization', wave=wavelet_name)
        self.dwtLx32 = DWTForward(J=3, mode='periodization', wave=wavelet_name)
        self.dwtLx16 = DWTForward(J=4, mode='periodization', wave=wavelet_name)
        self.dwtLx8 = DWTForward(J=5, mode='periodization', wave=wavelet_name)
        
        
        ##conv
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        self.featuresconv = conv_3x3_bn(3, input_channel, 2)
        
        block = InvertedResidual

        self.wavelet_conv = Block_lite_lite(in_channels = 3, out_channels = input_channel, is_supplement = True, wavelet_name = wavelet_name)
        lastchannel = input_channel
        ##features0
        output_channel = _make_divisible(self.cfgs[0][1] * width_mult, 4 if width_mult == 0.1 else 8)
        layer0 = []
        for i in range(self.cfgs[0][2]):
            layer0.append(block(input_channel, output_channel, self.cfgs[0][3] if i == 0 else 1, self.cfgs[0][0]))
            input_channel = output_channel
        self.features0 = nn.Sequential(*layer0)
        ##features1
        output_channel = _make_divisible(self.cfgs[1][1] * width_mult, 4 if width_mult == 0.1 else 8)
        layer1 = []
        for i in range(self.cfgs[1][2]):
            layer1.append(block(input_channel, output_channel, self.cfgs[1][3] if i == 0 else 1, self.cfgs[1][0]))
            input_channel = output_channel
        self.features1 = nn.Sequential(*layer1)        
        
        self.wavelet_1 = Block_lite_lite(in_channels = lastchannel, out_channels = output_channel, is_supplement = True, wavelet_name = wavelet_name)
        lastchannel = output_channel
        
        
        ##features2
        output_channel = _make_divisible(self.cfgs[2][1] * width_mult, 4 if width_mult == 0.1 else 8)
        layer2 = []
        for i in range(self.cfgs[2][2]):
            layer2.append(block(input_channel, output_channel, self.cfgs[2][3] if i == 0 else 1, self.cfgs[2][0]))
            input_channel = output_channel
        self.features2 = nn.Sequential(*layer2)        
        
        self.wavelet_2 = Block_lite_lite(in_channels = lastchannel, out_channels = output_channel, is_supplement = True, wavelet_name = wavelet_name)
        lastchannel = output_channel
        
        
        ##features3
        output_channel = _make_divisible(self.cfgs[3][1] * width_mult, 4 if width_mult == 0.1 else 8)
        layer3 = []
        for i in range(self.cfgs[3][2]):
            layer3.append(block(input_channel, output_channel, self.cfgs[3][3] if i == 0 else 1, self.cfgs[3][0]))
            input_channel = output_channel
        self.features3 = nn.Sequential(*layer3)        
        self.wavelet_3 = Block_lite_lite(in_channels = lastchannel, out_channels = output_channel, is_supplement = True, wavelet_name = wavelet_name)
        lastchannel = output_channel
        
        
        ##features4
        output_channel = _make_divisible(self.cfgs[4][1] * width_mult, 4 if width_mult == 0.1 else 8)
        layer4 = []
        for i in range(self.cfgs[4][2]):
            layer4.append(block(input_channel, output_channel, self.cfgs[4][3] if i == 0 else 1, self.cfgs[4][0]))
            input_channel = output_channel
        self.features4 = nn.Sequential(*layer4)        
        
        self.wavelet_4 = Block_lite_lite(in_channels = lastchannel, out_channels = output_channel, is_supplement = True, wavelet_name = wavelet_name)
        lastchannel = output_channel
        
        
        ##features5
        output_channel = _make_divisible(self.cfgs[5][1] * width_mult, 4 if width_mult == 0.1 else 8)
        layer5 = []
        for i in range(self.cfgs[5][2]):
            layer5.append(block(input_channel, output_channel, self.cfgs[5][3] if i == 0 else 1, self.cfgs[5][0]))
            input_channel = output_channel
        self.features5 = nn.Sequential(*layer5)        
        
        self.wavelet_5 = Block_lite_lite(in_channels = lastchannel, out_channels = output_channel, is_supplement = True, wavelet_name = wavelet_name)
        lastchannel = output_channel
        
        
        ##features6
        output_channel = _make_divisible(self.cfgs[6][1] * width_mult, 4 if width_mult == 0.1 else 8)
        layer6 = []
        for i in range(self.cfgs[6][2]):
            layer6.append(block(input_channel, output_channel, self.cfgs[6][3] if i == 0 else 1, self.cfgs[6][0]))
            input_channel = output_channel
        self.features6 = nn.Sequential(*layer6)
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)
        
        self._initialize_weights()

    def forward(self, x):
        
        Lx128,Yh = self.dwtLx128(x)
        Lx64,Yh = self.dwtLx64(x)
        Lx32,Yh = self.dwtLx32(x)
        Lx16,Yh = self.dwtLx16(x)
        Lx8,Yh = self.dwtLx8(x)
        
        x = self.featuresconv(x)
        x , Lnext = self.wavelet_conv(Llast = Lx128, original = Lx128, x = x, is_downsample = True)
        x = self.features0(x)
        x = self.features1(x)
        x , Lnext = self.wavelet_1(Llast = Lnext, original = Lx64, x = x, is_downsample = True)
        x = self.features2(x)
        x , Lnext = self.wavelet_2(Llast = Lnext, original = Lx32, x = x, is_downsample = True)
        x = self.features3(x)
        x , Lnext = self.wavelet_3(Llast = Lnext, original = Lx16, x = x, is_downsample = False)
        x = self.features4(x)
        x , Lnext = self.wavelet_4(Llast = Lnext, original = Lx16, x = x, is_downsample = True)
        x = self.features5(x)
        x , Lnext = self.wavelet_5(Llast = Lnext, original = Lx8, x = x, is_downsample = True)

        x = self.features6(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
def mobilenetv2(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    # return MobileNetV2(**kwargs)
    return MobileNetV2_wavelet(**kwargs)

