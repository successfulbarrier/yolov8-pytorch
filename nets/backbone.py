import torch
import torch.nn as nn
from .bottleneck_transformer_pytorch import BottleStack

def autopad(k, p=None, d=1):  
    # kernel, padding, dilation
    # 对输入的特征层进行自动padding，按照Same原则
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class SiLU(nn.Module):  
    # SiLU激活函数
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
    
class Conv(nn.Module):
    # 标准卷积+标准化+激活函数
    default_act = SiLU() 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act    = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    # 标准瓶颈结构，残差结构
    # c1为输入通道数，c2为输出通道数
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))    

class C2f(nn.Module):
    # CSPNet结构结构，大残差结构
    # c1为输入通道数，c2为输出通道数
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c      = int(c2 * e) 
        self.cv1    = Conv(c1, 2 * self.c, 1, 1)
        self.cv2    = Conv((2 + n) * self.c, c2, 1)
        self.m      = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        # 进行一个卷积，然后划分成两份，每个通道都为c
        y = list(self.cv1(x).split((self.c, self.c), 1))
        # 每进行一次残差结构都保留，然后堆叠在一起，密集残差
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
class SPPF(nn.Module):
    # SPP结构，5、9、13最大池化核的最大池化。
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_          = c1 // 2
        self.cv1    = Conv(c1, c_, 1, 1)
        self.cv2    = Conv(c_ * 4, c2, 1, 1)
        self.m      = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class Backbone(nn.Module):
    def __init__(self, base_channels, base_depth, deep_mul, phi, pretrained=False):
        super().__init__()
        #-----------------------------------------------#
        #   输入图片是3, 640, 640
        #-----------------------------------------------#
        # 3, 640, 640 => 32, 640, 640 => 64, 320, 320
        self.stem = Conv(3, base_channels, 3, 2)
        
        # 64, 320, 320 => 128, 160, 160 => 128, 160, 160
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),
            C2f(base_channels * 2, base_channels * 2, base_depth, True),
        )
        # 128, 160, 160 => 256, 80, 80 => 256, 80, 80
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C2f(base_channels * 4, base_channels * 4, base_depth * 2, True),
        )
        # 256, 80, 80 => 512, 40, 40 => 512, 40, 40
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C2f(base_channels * 8, base_channels * 8, base_depth * 2, True),
        )
        # 512, 40, 40 => 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 20, 20
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2),
            C2f(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth, True),
            SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), k=5)
        )
        
        if pretrained:
            url = {
                "n" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_n_backbone_weights.pth',
                "s" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_s_backbone_weights.pth',
                "m" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_m_backbone_weights.pth',
                "l" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_l_backbone_weights.pth',
                "x" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_x_backbone_weights.pth',
            }[phi]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
            self.load_state_dict(checkpoint, strict=False)
            print("Load weights from " + url.split('/')[-1])

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        #-----------------------------------------------#
        #   dark3的输出为256, 80, 80，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark3(x)
        feat1 = x
        #-----------------------------------------------#
        #   dark4的输出为512, 40, 40，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark4(x)
        feat2 = x
        #-----------------------------------------------#
        #   dark5的输出为1024 * deep_mul, 20, 20，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3

#-------------------------------------------------#
#   坐标注意力
#-------------------------------------------------#
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
    
#-------------------------------------------------#
#   dct特征和RGB特征融合模块,只是进行简单的上采样对齐RGB和DCT
#-------------------------------------------------#
class Backbone_RGB_DCT_fusion1(nn.Module):
    def __init__(self, base_channels, base_depth, deep_mul, phi, pretrained=False):
        super().__init__()
        #-----------------------------------------------#
        #   频域特征融合部分
        #-----------------------------------------------#
        # RGB特征提取 3, 640, 640 => 32, 640, 640 => 64, 320, 320
        self.conv1 = Conv(3, 32, 3, 2)
        self.conv2 = Conv(32, 64, 3, 2)
        # 频域特征
        # 定义一个二倍上采样层,输入是192通道80*80
        self.dct_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        #-------------------------------------------------#
        #   下面是正常的YOLOV8主干
        #-------------------------------------------------#
        # 64, 320, 320 => 128, 160, 160 => 128, 160, 160
        self.dark2 = C2f(256, base_channels * 2, base_depth, True)

        # 128, 160, 160 => 256, 80, 80 => 256, 80, 80
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C2f(base_channels * 4, base_channels * 4, base_depth * 2, True),
        )
        # 256, 80, 80 => 512, 40, 40 => 512, 40, 40
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C2f(base_channels * 8, base_channels * 8, base_depth * 2, True),
        )
        # 512, 40, 40 => 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 20, 20
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2),
            C2f(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth, True),
            SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), k=5)
        )
        
        if pretrained:
            url = {
                "n" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_n_backbone_weights.pth',
                "s" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_s_backbone_weights.pth',
                "m" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_m_backbone_weights.pth',
                "l" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_l_backbone_weights.pth',
                "x" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_x_backbone_weights.pth',
            }[phi]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
            self.load_state_dict(checkpoint, strict=False)
            print("Load weights from " + url.split('/')[-1])

    def forward(self, x, dct):
        #-------------------------------------------------#
        #   RGB特征和频域特征融合
        #-------------------------------------------------#
        x = self.conv1(x)
        x = self.conv2(x)
        dct = self.dct_upsample(dct)
        # 在通道维度上拼接x和dct
        x = torch.cat([x, dct], dim=1)
        
        #-------------------------------------------------#
        #   正常的YOLOV8特征提取
        #-------------------------------------------------#
        x = self.dark2(x)
        #-----------------------------------------------#
        #   dark3的输出为256, 80, 80，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark3(x)
        feat1 = x
        #-----------------------------------------------#
        #   dark4的输出为512, 40, 40，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark4(x)
        feat2 = x
        #-----------------------------------------------#
        #   dark5的输出为1024 * deep_mul, 20, 20，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3


#-------------------------------------------------#
#   dct特征和RGB特征融合模块,只是进行简单的上采样对齐RGB和DCT
#-------------------------------------------------#
class Backbone_RGB_DCT_fusion2(nn.Module):
    def __init__(self, base_channels, base_depth, deep_mul, phi, pretrained=False):
        super().__init__()
        #-----------------------------------------------#
        #   频域特征融合部分
        #-----------------------------------------------#
        # RGB特征提取 3, 640, 640 => 32, 640, 640 => 64, 320, 320
        self.conv1 = Conv(3, 32, 3, 2)
        self.conv2 = Conv(32, 64, 3, 2)
        # 频域特征
        # 定义一个二倍上采样层,输入是192通道80*80
        self.dct_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.MHSA_H = BottleStack(
                        dim = 96,               # channels in
                        fmap_size = 80,         # feature map size
                        dim_out = 96,           # channels out
                        proj_factor = 4,        # projection factor
                        downsample = False,     # downsample on first layer or not
                        heads = 3,              # number of heads
                        dim_head = 128,         # dimension per head, defaults to 128
                        rel_pos_emb = False,    # use relative positional embedding - uses absolute if False
                        activation = nn.ReLU()  # activation throughout the network
                    )
        self.MHSA_L = BottleStack(
                        dim = 96,               # channels in
                        fmap_size = 80,         # feature map size
                        dim_out = 96,           # channels out
                        proj_factor = 4,        # projection factor
                        downsample = False,     # downsample on first layer or not
                        heads = 3,              # number of heads
                        dim_head = 128,         # dimension per head, defaults to 128
                        rel_pos_emb = False,    # use relative positional embedding - uses absolute if False
                        activation = nn.ReLU()  # activation throughout the network
                    )
        self.MHSA_ALL = BottleStack(
                        dim = 192,              # channels in
                        fmap_size = 80,         # feature map size
                        dim_out = 192,          # channels out
                        proj_factor = 4,        # projection factor
                        downsample = False,     # downsample on first layer or not
                        heads = 3,              # number of heads
                        dim_head = 128,         # dimension per head, defaults to 128
                        rel_pos_emb = False,    # use relative positional embedding - uses absolute if False
                        activation = nn.ReLU()  # activation throughout the network
                    )
        
        self.coordAtt = CoordAtt(192)   # 主需要设置输出通道
        
        #-------------------------------------------------#
        #   下面是正常的YOLOV8主干
        #-------------------------------------------------#
        # 64, 320, 320 => 128, 160, 160 => 128, 160, 160
        self.dark2 = C2f(256, base_channels * 2, base_depth, True)

        # 128, 160, 160 => 256, 80, 80 => 256, 80, 80
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C2f(base_channels * 4, base_channels * 4, base_depth * 2, True),
        )
        # 256, 80, 80 => 512, 40, 40 => 512, 40, 40
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C2f(base_channels * 8, base_channels * 8, base_depth * 2, True),
        )
        # 512, 40, 40 => 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 20, 20
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2),
            C2f(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth, True),
            SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), k=5)
        )
        
        if pretrained:
            url = {
                "n" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_n_backbone_weights.pth',
                "s" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_s_backbone_weights.pth',
                "m" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_m_backbone_weights.pth',
                "l" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_l_backbone_weights.pth',
                "x" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_x_backbone_weights.pth',
            }[phi]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
            self.load_state_dict(checkpoint, strict=False)
            print("Load weights from " + url.split('/')[-1])

    def forward(self, x, dct):
        #-------------------------------------------------#
        #   RGB特征和频域特征融合
        #-------------------------------------------------#
        x = self.conv1(x)
        x = self.conv2(x)
        
        dct1, dct2 = torch.chunk(dct, 2, dim=1)
        dct1 = self.MHSA_L(dct1)
        dct2 = self.MHSA_H(dct2)
        dct = torch.cat([dct1, dct2], dim=1)
        dct = self.MHSA_ALL(dct)
        dct = self.coordAtt(dct)
        dct = self.dct_upsample(dct)
        # 在通道维度上拼接x和dct
        x = torch.cat([x, dct], dim=1)
        
        #-------------------------------------------------#
        #   正常的YOLOV8特征提取
        #-------------------------------------------------#
        x = self.dark2(x)
        #-----------------------------------------------#
        #   dark3的输出为256, 80, 80，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark3(x)
        feat1 = x
        #-----------------------------------------------#
        #   dark4的输出为512, 40, 40，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark4(x)
        feat2 = x
        #-----------------------------------------------#
        #   dark5的输出为1024 * deep_mul, 20, 20，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3

#-------------------------------------------------#
#   CBAM
#-------------------------------------------------#
class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))

#-------------------------------------------------#
#   使用CBAM替换BottleStack,不使用坐标注意力
#-------------------------------------------------#
class Backbone_RGB_DCT_fusion3(nn.Module):
    def __init__(self, base_channels, base_depth, deep_mul, phi, pretrained=False):
        super().__init__()
        #-----------------------------------------------#
        #   频域特征融合部分
        #-----------------------------------------------#
        # RGB特征提取 3, 640, 640 => 32, 640, 640 => 64, 320, 320
        self.conv1 = Conv(3, 96, 3, 2)
        self.conv2 = Conv(96, 192, 3, 2)
        # 频域特征
        # 定义一个二倍上采样层,输入是192通道80*80
        self.dct_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.CBAM_H = CBAM(96)
        self.CBAM_L = CBAM(96)
        self.CBAM_H_conv    = Conv(96, 32, 1, 1)
        self.CBAM_L_conv    = Conv(96, 32, 1, 1)
        self.CBAM_ALL = CBAM(64)
        
        # self.coordAtt = CoordAtt(192)   # 主需要设置输出通道
        
        #-------------------------------------------------#
        #   下面是正常的YOLOV8主干
        #-------------------------------------------------#
        # 64, 320, 320 => 128, 160, 160 => 128, 160, 160
        self.dark2 = C2f(256, base_channels * 2, base_depth, True)

        # 128, 160, 160 => 256, 80, 80 => 256, 80, 80
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C2f(base_channels * 4, base_channels * 4, base_depth * 2, True),
        )
        # 256, 80, 80 => 512, 40, 40 => 512, 40, 40
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C2f(base_channels * 8, base_channels * 8, base_depth * 2, True),
        )
        # 512, 40, 40 => 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 20, 20
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2),
            C2f(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth, True),
            SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), k=5)
        )
        
        if pretrained:
            url = {
                "n" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_n_backbone_weights.pth',
                "s" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_s_backbone_weights.pth',
                "m" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_m_backbone_weights.pth',
                "l" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_l_backbone_weights.pth',
                "x" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_x_backbone_weights.pth',
            }[phi]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
            self.load_state_dict(checkpoint, strict=False)
            print("Load weights from " + url.split('/')[-1])

    def forward(self, x, dct):
        #-------------------------------------------------#
        #   RGB特征和频域特征融合
        #-------------------------------------------------#
        x = self.conv1(x)
        x = self.conv2(x)
        
        dct1, dct2 = torch.chunk(dct, 2, dim=1)
        dct1 = self.CBAM_L(dct1)
        dct1 = self.CBAM_L_conv(dct1)
        dct2 = self.CBAM_H(dct2)
        dct2 = self.CBAM_H_conv(dct2)
        dct = torch.cat([dct1, dct2], dim=1)
        dct = self.CBAM_ALL(dct)
        # dct = self.coordAtt(dct)
        dct = self.dct_upsample(dct)
        # 在通道维度上拼接x和dct
        x = torch.cat([x, dct], dim=1)
        
        #-------------------------------------------------#
        #   正常的YOLOV8特征提取
        #-------------------------------------------------#
        x = self.dark2(x)
        #-----------------------------------------------#
        #   dark3的输出为256, 80, 80，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark3(x)
        feat1 = x
        #-----------------------------------------------#
        #   dark4的输出为512, 40, 40，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark4(x)
        feat2 = x
        #-----------------------------------------------#
        #   dark5的输出为1024 * deep_mul, 20, 20，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3


#-------------------------------------------------#
#   使用CBAM替换BottleStack，使用坐标注意力
#-------------------------------------------------#
class Backbone_RGB_DCT_fusion4(nn.Module):
    def __init__(self, base_channels, base_depth, deep_mul, phi, pretrained=False):
        super().__init__()
        #-----------------------------------------------#
        #   频域特征融合部分
        #-----------------------------------------------#
        # RGB特征提取 3, 640, 640 => 32, 640, 640 => 64, 320, 320
        self.conv1 = Conv(3, 96, 3, 2)
        self.conv2 = Conv(96, 192, 3, 2)
        # 频域特征
        # 定义一个二倍上采样层,输入是192通道80*80
        self.dct_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.CBAM_H = CBAM(96)
        self.CBAM_L = CBAM(96)
        self.CBAM_H_conv    = Conv(96, 32, 1, 1)
        self.CBAM_L_conv    = Conv(96, 32, 1, 1)
        self.CBAM_ALL = CBAM(64)
        
        self.coordAtt = CoordAtt(64)   # 主需要设置输出通道
        
        #-------------------------------------------------#
        #   下面是正常的YOLOV8主干
        #-------------------------------------------------#
        # 64, 320, 320 => 128, 160, 160 => 128, 160, 160
        self.dark2 = C2f(256, base_channels * 2, base_depth, True)

        # 128, 160, 160 => 256, 80, 80 => 256, 80, 80
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C2f(base_channels * 4, base_channels * 4, base_depth * 2, True),
        )
        # 256, 80, 80 => 512, 40, 40 => 512, 40, 40
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C2f(base_channels * 8, base_channels * 8, base_depth * 2, True),
        )
        # 512, 40, 40 => 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 20, 20
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2),
            C2f(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth, True),
            SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), k=5)
        )
        
        if pretrained:
            url = {
                "n" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_n_backbone_weights.pth',
                "s" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_s_backbone_weights.pth',
                "m" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_m_backbone_weights.pth',
                "l" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_l_backbone_weights.pth',
                "x" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_x_backbone_weights.pth',
            }[phi]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
            self.load_state_dict(checkpoint, strict=False)
            print("Load weights from " + url.split('/')[-1])

    def forward(self, x, dct):
        #-------------------------------------------------#
        #   RGB特征和频域特征融合
        #-------------------------------------------------#
        x = self.conv1(x)
        x = self.conv2(x)
        
        dct1, dct2 = torch.chunk(dct, 2, dim=1)
        dct1 = self.CBAM_L(dct1)
        dct1 = self.CBAM_L_conv(dct1)
        dct2 = self.CBAM_H(dct2)
        dct2 = self.CBAM_H_conv(dct2)
        dct = torch.cat([dct1, dct2], dim=1)
        dct = self.CBAM_ALL(dct)
        dct = self.coordAtt(dct)
        dct = self.dct_upsample(dct)
        # 在通道维度上拼接x和dct
        x = torch.cat([x, dct], dim=1)
        
        #-------------------------------------------------#
        #   正常的YOLOV8特征提取
        #-------------------------------------------------#
        x = self.dark2(x)
        #-----------------------------------------------#
        #   dark3的输出为256, 80, 80，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark3(x)
        feat1 = x
        #-----------------------------------------------#
        #   dark4的输出为512, 40, 40，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark4(x)
        feat2 = x
        #-----------------------------------------------#
        #   dark5的输出为1024 * deep_mul, 20, 20，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3


#-------------------------------------------------#
#   对不同频率的dct乘以不同的权重系数
#-------------------------------------------------#
class ChannelSplitModule(nn.Module):
    def __init__(self, input_channels, feat_size, split_num = 6):
        super(ChannelSplitModule, self).__init__()
        self.num_channels = input_channels
        self.split_channels = input_channels // split_num
        self.split_num = split_num
        self.weights = nn.Parameter(torch.randn(split_num, feat_size, feat_size))
        self.dctl = [1, 2 , 3, 4, 5, 6]
        self.bn = nn.BatchNorm2d(input_channels, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        
    def forward(self, x):
        # 将输入数据从通道维度上分成5部分
        x = self.bn(x)
        split_x = torch.chunk(x, self.split_num, dim=1)
        # 初始化结果
        result = []
        for i, x_split in enumerate(split_x):
            # 和对应的参数矩阵相乘
            result.append(x_split * torch.clamp(self.weights[i], 0, 1)*self.dctl[i])
        result = torch.cat(result, 1)
        return result


#-------------------------------------------------#
#   使用CBAM替换BottleStack，使用坐标注意力
#-------------------------------------------------#
class Backbone_RGB_DCT_fusion5(nn.Module):
    def __init__(self, base_channels, base_depth, deep_mul, phi, pretrained=False):
        super().__init__()
        #-----------------------------------------------#
        #   频域特征融合部分
        #-----------------------------------------------#
        # RGB特征提取 3, 640, 640 => 32, 640, 640 => 64, 320, 320
        self.conv1 = Conv(3, 96, 3, 2)
        self.conv2 = Conv(96, 192, 3, 2)
        # 频域特征
        # 定义一个二倍上采样层,输入是192通道80*80
        self.channel_split_module = ChannelSplitModule(input_channels=192, feat_size=80)
        self.dct_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.CBAM_H = CBAM(96)
        self.CBAM_L = CBAM(96)
        self.CBAM_H_conv    = Conv(96, 32, 1, 1)
        self.CBAM_L_conv    = Conv(96, 32, 1, 1)
        self.CBAM_ALL = CBAM(64)
        
        self.coordAtt = CoordAtt(64)   # 主需要设置输出通道
              
        #-------------------------------------------------#
        #   下面是正常的YOLOV8主干
        #-------------------------------------------------#
        # 64, 320, 320 => 128, 160, 160 => 128, 160, 160
        self.dark2 = C2f(256, base_channels * 2, base_depth, True)

        # 128, 160, 160 => 256, 80, 80 => 256, 80, 80
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C2f(base_channels * 4, base_channels * 4, base_depth * 2, True),
        )
        # 256, 80, 80 => 512, 40, 40 => 512, 40, 40
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C2f(base_channels * 8, base_channels * 8, base_depth * 2, True),
        )
        # 512, 40, 40 => 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 20, 20
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2),
            C2f(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth, True),
            SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), k=5)
        )
        
        if pretrained:
            url = {
                "n" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_n_backbone_weights.pth',
                "s" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_s_backbone_weights.pth',
                "m" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_m_backbone_weights.pth',
                "l" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_l_backbone_weights.pth',
                "x" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_x_backbone_weights.pth',
            }[phi]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
            self.load_state_dict(checkpoint, strict=False)
            print("Load weights from " + url.split('/')[-1])

    def forward(self, x, dct):
        #-------------------------------------------------#
        #   RGB特征和频域特征融合
        #-------------------------------------------------#
        x = self.conv1(x)
        x = self.conv2(x)
        
        dct = self.channel_split_module(dct)
        dct1, dct2 = torch.chunk(dct, 2, dim=1)
        dct1 = self.CBAM_L(dct1)
        dct1 = self.CBAM_L_conv(dct1)
        dct2 = self.CBAM_H(dct2)
        dct2 = self.CBAM_H_conv(dct2)
        dct = torch.cat([dct1, dct2], dim=1)
        dct = self.CBAM_ALL(dct)
        dct = self.coordAtt(dct)
        dct = self.dct_upsample(dct)
        # 在通道维度上拼接x和dct
        x = torch.cat([x, dct], dim=1)
        
        #-------------------------------------------------#
        #   正常的YOLOV8特征提取
        #-------------------------------------------------#
        x = self.dark2(x)
        #-----------------------------------------------#
        #   dark3的输出为256, 80, 80，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark3(x)
        feat1 = x
        #-----------------------------------------------#
        #   dark4的输出为512, 40, 40，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark4(x)
        feat2 = x
        #-----------------------------------------------#
        #   dark5的输出为1024 * deep_mul, 20, 20，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3

#-------------------------------------------------#
#   使用CBAM替换BottleStack，使用坐标注意力
#-------------------------------------------------#
class Backbone_RGB_DCT_fusion6(nn.Module):
    def __init__(self, base_channels, base_depth, deep_mul, phi, pretrained=False):
        super().__init__()
        #-----------------------------------------------#
        #   频域特征融合部分
        #-----------------------------------------------#
        # 定义一个二倍上采样层,输入是192通道80*80
        self.channel_split_module = ChannelSplitModule(input_channels=192, feat_size=80)
        self.dct_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.CBAM_H = CBAM(96)
        self.CBAM_L = CBAM(96)
        self.CBAM_H_conv    = Conv(96, 32, 1, 1)
        self.CBAM_L_conv    = Conv(96, 32, 1, 1)
        self.CBAM_ALL = CBAM(64)
        
        self.coordAtt = CoordAtt(64)   # 主需要设置输出通道
        
        # 拼接到20*20
        self.neck_conv20    = Conv(96, 128, 3, 2)   
        # 拼接到40*40
        self.neck_conv40    = Conv(64, 96, 3, 2)
               
        #-------------------------------------------------#
        #   下面是正常的YOLOV8主干
        #-------------------------------------------------#
        # RGB特征提取 3, 640, 640 => 32, 640, 640 => 64, 320, 320
        self.stem = Conv(3, base_channels, 3, 2)
        
        # 64, 320, 320 => 128, 160, 160 => 128, 160, 160
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),
            C2f(base_channels * 2, base_channels * 2, base_depth, True),
        )

        # 128, 160, 160 => 256, 80, 80 => 256, 80, 80
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C2f(base_channels * 4, base_channels * 4, base_depth * 2, True),
        )
        # 256, 80, 80 => 512, 40, 40 => 512, 40, 40
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C2f(base_channels * 8, base_channels * 8, base_depth * 2, True),
        )
        # 512, 40, 40 => 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 20, 20
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2),
            C2f(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth, True),
            SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), k=5)
        )
        
        if pretrained:
            url = {
                "n" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_n_backbone_weights.pth',
                "s" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_s_backbone_weights.pth',
                "m" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_m_backbone_weights.pth',
                "l" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_l_backbone_weights.pth',
                "x" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_x_backbone_weights.pth',
            }[phi]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
            self.load_state_dict(checkpoint, strict=False)
            print("Load weights from " + url.split('/')[-1])

    def forward(self, x, dct):
        #-------------------------------------------------#
        #   RGB特征和频域特征融合
        #-------------------------------------------------#
        dct = self.channel_split_module(dct)
        dct1, dct2 = torch.chunk(dct, 2, dim=1)
        dct1 = self.CBAM_L(dct1)
        dct1 = self.CBAM_L_conv(dct1)
        dct2 = self.CBAM_H(dct2)
        dct2 = self.CBAM_H_conv(dct2)
        dct = torch.cat([dct1, dct2], dim=1)
        dct = self.CBAM_ALL(dct)
        dct_feat1 = self.coordAtt(dct)
        dct_feat2 = self.neck_conv40(dct_feat1)
        dct_feat3 = self.neck_conv20(dct_feat2)

        #-------------------------------------------------#
        #   正常的YOLOV8特征提取
        #-------------------------------------------------#
        x = self.stem(x)
        x = self.dark2(x)
        #-----------------------------------------------#
        #   dark3的输出为256, 80, 80，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark3(x)
        feat1 = torch.cat([x, dct_feat1], dim=1)    # dct_feat1 64
        #-----------------------------------------------#
        #   dark4的输出为512, 40, 40，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark4(x)
        feat2 = torch.cat([x, dct_feat2], dim=1)    # dct_feat2 96
        #-----------------------------------------------#
        #   dark5的输出为1024 * deep_mul, 20, 20，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark5(x)
        feat3 = torch.cat([x, dct_feat3], dim=1)    # dct_feat3 128
        return feat1, feat2, feat3
