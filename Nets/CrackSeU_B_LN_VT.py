import torch.nn as nn
import torch
from torch import autograd
from functools import partial
import torch.nn.functional as F
from torchvision import models
import numpy as np


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class GELU(nn.Module):
    def __init__(self, inplace=True):
        super(GELU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return 0.5 * x * (1. + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x,3))))


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(3, 1, kernel_size, padding=padding, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True) 
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out2 = torch.mean(x, dim=1, keepdim=True)

        x = torch.cat([avg_out, max_out, avg_out2], dim=1) 
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False) 
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False) 
        self.sigmoid = nn.Sigmoid()
        self.GELU = GELU()

    def forward(self, x):
        avg_out = self.fc2(self.GELU(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.GELU(self.fc1(self.max_pool(x))))
        avg_out2 = self.fc2(self.GELU(self.fc1(self.avg_pool(x))))

        out = avg_out + max_out + avg_out2 
        return self.sigmoid(out)


class Stage1(nn.Module): # Fist Block
    def __init__(self, in_ch, out_ch, Active):
        super(Stage1, self).__init__()

        if Active == 'ReLU':
            Active_Layer = nn.ReLU
        elif Active == 'GELU':
            Active_Layer = GELU
        elif Active == 'Hswish':
            Active_Layer = Hswish

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride=1, padding=1)
        self.norm1 = LayerNorm(out_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False), 
            Active_Layer(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x) # (N, C, H, W)
        x = self.pool1(x)
        x = self.conv2(x)
        return x


class DoubleConv(nn.Module): 
    def __init__(self, in_ch, out_ch, Active):
        super(DoubleConv, self).__init__()

        if Active == 'ReLU':
            Active_Layer = nn.ReLU
        elif Active == 'GELU':
            Active_Layer = GELU
        elif Active == 'Hswish':
            Active_Layer = Hswish

        self.conv1= nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = LayerNorm(in_ch)
        self.conv2= nn.Sequential( # 3*3
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False), 
            Active_Layer(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        return x


class Upsample_block(nn.Module): 
    def __init__(self, in_ch, out_ch, s_factor):
        super(Upsample_block, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm = LayerNorm(out_ch) 
        self.act = Hswish(inplace=True)
        self.upsample = nn.Upsample(scale_factor=s_factor, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.upsample(x)
        return x


class Fusion_module(nn.Module): 
    def __init__(self, in_ch, out_ch, Active):
        super(Fusion_module, self).__init__()
        
        if Active == 'ReLU':
            Active_Layer = nn.ReLU
        elif Active == 'GELU':
            Active_Layer = GELU
        elif Active == 'Hswish':
            Active_Layer = Hswish

        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm1 = LayerNorm(in_ch)
        self.act1 = Active_Layer(inplace=True)
        
        self.conv2 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch, bias=False)
        self.norm2 = LayerNorm(in_ch)

        self.conv3 = nn.Conv2d(in_ch, in_ch//2, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm3 = LayerNorm(in_ch//2)
        self.act3 = Active_Layer(inplace=True)
        
        self.conv4 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch, bias=False)
        self.norm4 = LayerNorm(in_ch)
        
        self.conv5 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm5 = LayerNorm(out_ch)
        self.act5 = Active_Layer(inplace=True)
        
        self.ca = ChannelAttention(in_ch//2)
        self.sa = SpatialAttention()
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        res = x
        x = self.conv2(x)
        x = self.norm2(x)
        
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.act3(x)

        out1 = self.ca(x)
        out2 = self.sa(x)
        out1 = out1 * x
        out2 = out2 * x
        out = torch.cat([out1, out2], dim=1)

        out = out + res
        out = self.conv4(out)
        out = self.norm4(out)
        
        out = self.conv5(out)
        out = self.norm5(out)
        out = self.act5(out)

        return out


class Side_output_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Side_output_block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1)
        self.norm1 = LayerNorm(64)
        self.conv2 = nn.Conv2d(64, out_ch, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        return x


class Output_block(nn.Module): 
    def __init__(self, in_ch, out_ch):
        super(Output_block, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_ch, 32, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, out_ch, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class CrackSeU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CrackSeU, self).__init__()

        filter = [32, 64, 128, 256, 512] 

        # ------------- Encoder, Downsample Block -------------- #
        self.conv1 = Stage1(in_ch, filter[0],'GELU')  # /2
        self.pool1 = nn.MaxPool2d(2) # /4
        self.conv2 = DoubleConv(filter[0], filter[1],'GELU') 
        self.pool2 = nn.MaxPool2d(2) # /8
        self.conv3 = DoubleConv(filter[1], filter[2],'GELU') 
        self.pool3 = nn.MaxPool2d(2) # /16
        self.conv4 = DoubleConv(filter[2], filter[3],'Hswish') 
        self.pool4 = nn.MaxPool2d(2) # /32
        self.conv5 = DoubleConv(filter[3], filter[4],'Hswish') 

        # ------------- Upsample Block -------------- #
        self.upsample_5_3 = Upsample_block(512, 256, 4)
        self.upsample_4_3 = Upsample_block(256, 128, 2)
        self.upsample_3_1 = Upsample_block(128*2, 128, 4)
        self.upsample_2_1 = Upsample_block(64, 32, 2)

        # ------------- Feature Fusion Module -------------- #
        self.fusion1 = Fusion_module(128*4, 128*2,'Hswish') 
        self.fusion2 = Fusion_module(64, 64,'Hswish')
        self.fusion3 = Fusion_module(192, 32,'Hswish') 

        # ------------- Side Output Block -------------- #
        self.min_2x = Side_output_block(128, out_ch)

        # ------------- Output Block -------------- #
        self.upsample_1_0 = Output_block(32, out_ch)

    def forward(self, x):
        
        # ------------- Downsample -------------- #
        c1 = self.conv1(x) 
        p1 = self.pool1(c1) # print(p1.shape)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        
        # ------------- Feature Fusion -------------- #
        up_5_3 = self.upsample_5_3(c5)
        up_4_3 = self.upsample_4_3(c4)
        merge1 = torch.cat([up_5_3, up_4_3, c3], dim=1)
        fusion1_out = self.fusion1(merge1)

        up_2_1 = self.upsample_2_1(c2)
        merge2 = torch.cat([up_2_1, c1], dim=1)
        fusion2_out = self.fusion2(merge2)

        up_3_1 = self.upsample_3_1(fusion1_out)
        merge3 = torch.cat([up_3_1, fusion2_out], dim=1)
        fusion3_out = self.fusion3(merge3)

        # ------------- Define Output -------------- #
        up_1_0 = self.upsample_1_0(fusion3_out)
        
        # ------------- Side Output -------------- #
        min_2x_out = self.min_2x(up_3_1)
        min_2x_out = nn.Sigmoid()(min_2x_out)

        # ------------- Output -------------- #
        out = nn.Sigmoid()(up_1_0)
        return out, min_2x_out


class CrackSeU_Inference(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CrackSeU_Inference, self).__init__()

        filter = [32, 64, 128, 256, 512] 

        # ------------- Encoder, Downsample Block -------------- #
        self.conv1 = Stage1(in_ch, filter[0],'GELU')  # /2
        self.pool1 = nn.MaxPool2d(2) # /4
        self.conv2 = DoubleConv(filter[0], filter[1],'GELU') 
        self.pool2 = nn.MaxPool2d(2) # /8
        self.conv3 = DoubleConv(filter[1], filter[2],'GELU') 
        self.pool3 = nn.MaxPool2d(2) # /16
        self.conv4 = DoubleConv(filter[2], filter[3],'Hswish') 
        self.pool4 = nn.MaxPool2d(2) # /32
        self.conv5 = DoubleConv(filter[3], filter[4],'Hswish') 

        # ------------- Upsample Block -------------- #
        self.upsample_5_3 = Upsample_block(512, 256, 4)
        self.upsample_4_3 = Upsample_block(256, 128, 2)
        self.upsample_3_1 = Upsample_block(128*2, 128, 4)
        self.upsample_2_1 = Upsample_block(64, 32, 2)

        # ------------- Feature Fusion Module -------------- #
        self.fusion1 = Fusion_module(128*4, 128*2,'Hswish') 
        self.fusion2 = Fusion_module(64, 64,'Hswish')
        self.fusion3 = Fusion_module(192, 32,'Hswish')

        # ------------- Output Block -------------- #
        self.upsample_1_0 = Output_block(32, out_ch)

    def forward(self, x):
        
        # ------------- Downsample -------------- #
        c1 = self.conv1(x) 
        p1 = self.pool1(c1) # print(p1.shape)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        
        # ------------- Feature Fusion -------------- #
        up_5_3 = self.upsample_5_3(c5)
        up_4_3 = self.upsample_4_3(c4)
        merge1 = torch.cat([up_5_3, up_4_3, c3], dim=1)
        fusion1_out = self.fusion1(merge1)

        up_2_1 = self.upsample_2_1(c2)
        merge2 = torch.cat([up_2_1, c1], dim=1)
        fusion2_out = self.fusion2(merge2)

        up_3_1 = self.upsample_3_1(fusion1_out)
        merge3 = torch.cat([up_3_1, fusion2_out], dim=1)
        fusion3_out = self.fusion3(merge3)

        # ------------- Define Output -------------- #
        up_1_0 = self.upsample_1_0(fusion3_out)

        # ------------- Output -------------- #
        out = nn.Sigmoid()(up_1_0)
        return out