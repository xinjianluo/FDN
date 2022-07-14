import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import pytorch_ssim
import pickle
from sklearn.cluster import KMeans
import random
import torchvision.utils as tvutils
import numpy as np
import torchvision.models as tvmodels
import RNANcommon  
     
########################################## Fusion Net ##########################################

class ConvBlock(nn.Module):
    def __init__(self, inplane, outplane, stride=1):
        super(ConvBlock, self).__init__()
        self.padding = (1, 1, 1, 1)
        self.conv = nn.Conv2d(inplane, outplane, kernel_size=3, padding=0, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(outplane)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = F.pad(x, self.padding, 'replicate')
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out

class FusionNet(nn.Module):
    def __init__(self, n_colors=3, fuse_scheme=0):
        super(FusionNet, self).__init__()  
        self.fuse_scheme = fuse_scheme # MAX 0, MEAN 2
        kernel_num = 64
        # image relaxing (registration)
        self.redconv1 = nn.Sequential(nn.Conv2d(n_colors, kernel_num, kernel_size=3, stride=2, padding=1), 
                                   nn.ReLU()) 
        self.deconv1 = nn.ConvTranspose2d(kernel_num, kernel_num, kernel_size=3, stride=2, padding=1)

        self.conv2 = ConvBlock(kernel_num, kernel_num)
        self.conv3 = ConvBlock(kernel_num, kernel_num)
        self.conv4 = nn.Conv2d(kernel_num, 3, kernel_size=1, padding=0, stride=1, bias=True)

        # Initialize parameters for other parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def tensor_max(self, tensors):
        max_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                max_tensor = tensor
            else:
                max_tensor = torch.max(max_tensor, tensor)
        return max_tensor

    def tensor_mean(self, tensors):
        sum_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                sum_tensor = tensor
            else:
                sum_tensor = sum_tensor + tensor
        mean_tensor = sum_tensor / len(tensors)
        return mean_tensor

    def operate(self, operator, tensors, doublesize = False):
        out_tensors = []
        for tensor in tensors:
            if doublesize:
                b, c, h, w = tensor.size()
                out_tensor = operator(tensor, output_size=(b, c, h*2, w*2))
            else:
                out_tensor = operator(tensor)
            out_tensors.append(out_tensor)
        return out_tensors
        
    def forward(self, *tensors):
        # Feature extraction
        outs = tensors
        outs = self.operate(self.redconv1, outs) 
        outs = self.operate(self.deconv1, outs, doublesize=True) 
        outs = self.operate(nn.ReLU(), outs) 
       
        outs = self.operate(self.conv2, outs)
        
        # Feature fusion
        if self.fuse_scheme == 0: # MAX
            out = self.tensor_max(outs)
        elif self.fuse_scheme == 2: # MEAN
            out = self.tensor_mean(outs)
        else: # Default: MAX
            out = self.tensor_max(outs)
        # Feature reconstruction
        out = self.conv3(out)
        out = self.conv4(out)
        return out

########################################## RNAN (Denoising) ##########################################

class _ResGroup(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, act):
        super(_ResGroup, self).__init__()
        modules_body = []
        modules_body.append(RNANcommon.ResAttModuleDownUpPlus(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        return res

class _NLResGroup(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, act):
        super(_NLResGroup, self).__init__()
        modules_body = []
        modules_body.append(RNANcommon.NLResAttModuleDownUpPlus(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        return res

class RNAN(nn.Module):
    def __init__(self, n_resgroup=10, n_feats=64, reduction=16, n_colors=3, conv=RNANcommon.default_conv):
        super(RNAN, self).__init__()
        kernel_size = 3
        act = nn.ReLU(True)
        
        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]
        
        # define body module
        modules_body_nl_low = [
            _NLResGroup(
                conv, n_feats, kernel_size, act=act)]
        modules_body = [
            _ResGroup(
                conv, n_feats, kernel_size, act=act) \
            for _ in range(n_resgroup - 2)]
        modules_body_nl_high = [
            _NLResGroup(
                conv, n_feats, kernel_size, act=act)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            conv(n_feats, n_colors, kernel_size)]
        self.head = nn.Sequential(*modules_head)
        self.body_nl_low = nn.Sequential(*modules_body_nl_low)
        self.body = nn.Sequential(*modules_body)
        self.body_nl_high = nn.Sequential(*modules_body_nl_high)
        self.tail = nn.Sequential(*modules_tail)
        
        self.convmerge = nn.Sequential(nn.Conv2d(n_colors, 3, 1)) 

    def forward(self, x):
        feats_shallow = self.head(x)

        res = self.body_nl_low(feats_shallow)
        res = self.body(res)
        res = self.body_nl_high(res)

        res_main = self.tail(res)
        res_clean = x + res_main

        return self.convmerge(res_clean) 

########################################## Comparative Network ##########################################
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x): 
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out
    
class ComparativeNN(nn.Module):
    def __init__(self, n_colors=3):
        super(ComparativeNN, self).__init__()  
        kernel_num = 96
        self.conv1 = ConvBlock(n_colors, kernel_num, stride=1) 
        self.residual_layer1 = self._make_layer(BasicBlock, kernel_num, 3, stride=1) 

        self.conv2 = ConvBlock(n_colors, kernel_num, stride=1) 
        self.residual_layer2 = self._make_layer(BasicBlock, kernel_num, 3, stride=1)  

        self.linear = nn.Sequential(nn.Linear(kernel_num*2, 2), nn.Sigmoid())
        self.cos = nn.CosineSimilarity(dim=2)
        self.cusprint = False
  
        # Initialize parameters for other parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, *tensors):
        outs = tensors

        subouts = self.downsample(outs)
        subouts = self.operate(self.conv1, subouts)
        subouts = self.operate(self.residual_layer1, subouts)
        
        halfouts = self.halfcrop(outs)
        halfouts = self.operate(self.conv2, halfouts)
        halfouts = self.operate(self.residual_layer2, halfouts)

        subout1, subout2 = subouts
        halfout1, halfout2 = halfouts
        
        halfout1 = halfout1.flatten(2)
        halfout2 = halfout2.flatten(2)
        subout1 = subout1.flatten(2)
        subout2 = subout2.flatten(2)

        subcos = self.cos(subout1, subout2)
        halfcos = self.cos(halfout1, halfout2)
        if self.cusprint:
            print(subout1.shape, halfout1.shape)
            print(subcos.shape, halfcos.shape) 

        cos = torch.cat((subcos, halfcos), dim=1)
        out = self.linear(cos)
        return out 
        
    def operate(self, operator, tensors, doublesize = False):
        out_tensors = []
        for tensor in tensors:
            if doublesize:
                b, c, h, w = tensor.size()
                out_tensor = operator(tensor, output_size=(b, c, h*2, w*2))
            else:
                out_tensor = operator(tensor)
            out_tensors.append(out_tensor)
        return out_tensors
    
    def setprint(self, flag=True):
        self.cusprint = flag
    
    def halfcrop(self, tensors):
        out_tensors = []
        # b, c, w, h
        for tensor in tensors:
            out_tensor = tensor[:, :, 5:-4, 5:-4]
            out_tensors.append(out_tensor)
        return out_tensors
    
    def downsample(self, tensors):
        # [0, 2, 4, 6, 8, ...]
        a = [i for i in range(32) if i%2==0]
        out_tensors = []
        # b, c, w, h
        for tensor in tensors:
            out_tensor = tensor[:, :, a, :]
            out_tensor = out_tensor[:, :, :, a]
            out_tensors.append(out_tensor)
        return out_tensors
    

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(planes, planes, stride))
            planes = planes * block.expansion
        return nn.Sequential(*layers)