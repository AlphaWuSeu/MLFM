import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from pytorch_wavelets import DWT1DForward, DWT1DInverse,DWTForward, DWTInverse # or simply DWT1D, IDWT1D


__all__ = ['SENet', 'se_resnet_10','se_resnet_18', 'se_resnet_34', 'se_resnet_50', 'se_resnet_101',
           'se_resnet_152', 'se_resnet_10_wavelet','se_resnet_18_wavelet', 'se_resnet_34_wavelet', 'se_resnet_50_wavelet', 'se_resnet_101_wavelet',
                      'se_resnet_152_wavelet']

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
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
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        # if planes == 64:
        #     self.globalAvgPool = nn.AvgPool2d(56, stride=1)
        # elif planes == 128:
        #     self.globalAvgPool = nn.AvgPool2d(28, stride=1)
        # elif planes == 256:
        #     self.globalAvgPool = nn.AvgPool2d(14, stride=1)
        # elif planes == 512:
        #     self.globalAvgPool = nn.AvgPool2d(7, stride=1)
        if planes == 64:
            self.globalAvgPool = nn.AvgPool2d(64, stride=1)
        elif planes == 128:
            self.globalAvgPool = nn.AvgPool2d(32, stride=1)
        elif planes == 256:
            self.globalAvgPool = nn.AvgPool2d(16, stride=1)
        elif planes == 512:
            self.globalAvgPool = nn.AvgPool2d(8, stride=1)
        self.fc1 = nn.Linear(in_features=planes, out_features=round(planes / 16))
        self.fc2 = nn.Linear(in_features=round(planes / 16), out_features=planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        original_out = out
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = out * original_out

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        if planes == 64:
            self.globalAvgPool = nn.AvgPool2d(56, stride=1)
        elif planes == 128:
            self.globalAvgPool = nn.AvgPool2d(28, stride=1)
        elif planes == 256:
            self.globalAvgPool = nn.AvgPool2d(14, stride=1)
        elif planes == 512:
            self.globalAvgPool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(in_features=planes * 4, out_features=round(planes / 4))
        self.fc2 = nn.Linear(in_features=round(planes / 4), out_features=planes * 4)
        self.sigmoid = nn.Sigmoid()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        original_out = out
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0),out.size(1),1,1)
        out = out * original_out

        out += residual
        out = self.relu(out)

        return out


class SENet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(SENet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
class SENet_wavelet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(SENet_wavelet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        wavelet_name = "haar"
        self.dwtLx128 = DWTForward(J=1, mode='periodization', wave=wavelet_name)
        self.dwtLx64 = DWTForward(J=2, mode='periodization', wave=wavelet_name)
        self.dwtLx32 = DWTForward(J=3, mode='periodization', wave=wavelet_name)
        self.dwtLx16 = DWTForward(J=4, mode='periodization', wave=wavelet_name)
        self.dwtLx8 = DWTForward(J=5, mode='periodization', wave=wavelet_name)
        
        self.wavelet_block0 = Block_lite_lite(in_channels = 3, out_channels = 64, is_supplement = True, wavelet_name = wavelet_name)
        self.wavelet_blockpool = Block_lite_lite(in_channels = 64, out_channels = 64, is_supplement = True, wavelet_name = wavelet_name)
        self.wavelet_block1 = Block_lite_lite(in_channels = 64, out_channels = 64, is_supplement = True, wavelet_name = wavelet_name)
        self.wavelet_block2 = Block_lite_lite(in_channels = 64, out_channels = 128, is_supplement = True, wavelet_name = wavelet_name)
        self.wavelet_block3 = Block_lite_lite(in_channels = 128, out_channels = 256, is_supplement = True, wavelet_name = wavelet_name)
        self.wavelet_block4 = Block_lite_lite(in_channels = 256, out_channels = 512, is_supplement = True, wavelet_name = wavelet_name)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        Lx128,Yh = self.dwtLx128(x)
        Lx64,Yh = self.dwtLx64(x)
        Lx32,Yh = self.dwtLx32(x)
        Lx16,Yh = self.dwtLx16(x)
        Lx8,Yh = self.dwtLx8(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x, Lnext = self.wavelet_block0(Llast = Lx128, original = Lx128, x = x, is_downsample = True)
        
        x = self.maxpool(x)
        x, Lnext = self.wavelet_blockpool(Llast = Lnext, original = Lx64, x = x, is_downsample = False)

        x = self.layer1(x)
        x, Lnext = self.wavelet_block1(Llast = Lnext, original = Lx64, x = x, is_downsample = True)

        x = self.layer2(x)
        x, Lnext = self.wavelet_block2(Llast = Lnext, original = Lx32, x = x, is_downsample = True)

        x = self.layer3(x)
        x, Lnext = self.wavelet_block3(Llast = Lnext, original = Lx16, x = x, is_downsample = True)

        x = self.layer4(x)
        x, Lnext = self.wavelet_block4(Llast = Lnext, original = Lx8, x = x, is_downsample = False)

        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
def se_resnet_10(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SENet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model

def se_resnet_18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SENet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def se_resnet_34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SENet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def se_resnet_50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SENet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def se_resnet_101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SENet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def se_resnet_152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SENet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

def se_resnet_10_wavelet(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SENet_wavelet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model

def se_resnet_18_wavelet(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SENet_wavelet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def se_resnet_34_wavelet(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SENet_wavelet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def se_resnet_50_wavelet(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SENet_wavelet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def se_resnet_101_wavelet(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SENet_wavelet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def se_resnet_152_wavelet(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SENet_wavelet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model