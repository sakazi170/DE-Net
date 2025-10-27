
import torch.nn as nn

class DWSConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DWSConv3d, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels
        )
        # Pointwise convolution
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DWSConv3d_Dilated(nn.Module):
    """DWS Conv with dilation support"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(DWSConv3d_Dilated, self).__init__()

        # Depthwise convolution with dilation
        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False
        )

        # Pointwise convolution
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DDSCR(nn.Module):
    """
    Dilated Depthwise Separable Convolutional Residual block
    Combines efficiency of DWS with multi-scale receptive field of dilated convolutions
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DDSCR, self).__init__()

        padding_d1 = (kernel_size - 1) * 1 // 2
        padding_d2 = (kernel_size - 1) * 2 // 2

        # First DWS conv with dilation=1
        self.conv1 = DWSConv3d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=padding_d1)
        self.in1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act1 = nn.GELU()

        # Second DWS conv with dilation=2
        self.conv2 = DWSConv3d_Dilated(out_channels, out_channels,
                                       kernel_size=kernel_size,
                                       stride=1,
                                       padding=padding_d2,
                                       dilation=2)
        self.in2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act2 = nn.GELU()

        # Residual connection
        self.residual_conv = nn.Identity() if in_channels == out_channels else \
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        residual = self.residual_conv(x)

        out = self.conv1(x)
        out = self.in1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.in2(out)
        out = self.act2(out)

        return out + residual

class LEB_strided(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Use stride in the depthwise conv for efficient downsampling
        self.conv = DWSConv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(min(8, out_channels), out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.norm(self.conv(x)))
        return x


class LBB(nn.Module):
    def __init__(self, in_channels):
        super(LBB, self).__init__()
        mid_channels = in_channels // 4

        # Bottleneck structure with depthwise separable convolutions
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1)  # Dimension reduction
        self.norm1 = nn.InstanceNorm3d(mid_channels)

        self.conv2 = DWSConv3d(mid_channels, mid_channels)  # Spatial processing
        self.norm2 = nn.InstanceNorm3d(mid_channels)

        self.conv3 = nn.Conv3d(mid_channels, in_channels, kernel_size=1)  # Dimension restoration
        self.norm3 = nn.InstanceNorm3d(in_channels)

        self.act = nn.GELU()

    def forward(self, x):
        identity = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.act(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        out = out + identity
        return self.act(out)
