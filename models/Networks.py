
import torch
import torch.nn as nn
import thop
import os
import sys

from modules.normal_blocks import DDSCR, LBB, LEB_strided
from modules.pgam import PGAM, PGAM1, PGAM2
from modules.aemf import AEMF, AEMF1, AEMF2, AEMF_NoEntropy
from modules.dme import DME, DME1, DME2, DME3

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


class base(nn.Module):
    def __init__(self, img_h, img_w, img_d, in_channels=1, num_classes=4):
        super().__init__()

        # Max pooling layer (shared across all branches)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder blocks for T1
        self.enc1_t1 = DDSCR(in_channels, 12)
        self.enc2_t1 = DDSCR(12, 24)
        self.enc3_t1 = DDSCR(24, 48)
        self.enc4_t1 = DDSCR(48, 96)

        # Encoder blocks for T1CE
        self.enc1_t1ce = DDSCR(in_channels, 12)
        self.enc2_t1ce = DDSCR(12, 24)
        self.enc3_t1ce = DDSCR(24, 48)
        self.enc4_t1ce = DDSCR(48, 96)

        # Encoder blocks for T2
        self.enc1_t2 = DDSCR(in_channels, 12)
        self.enc2_t2 = DDSCR(12, 24)
        self.enc3_t2 = DDSCR(24, 48)
        self.enc4_t2 = DDSCR(48, 96)

        # Encoder blocks for FLAIR
        self.enc1_flair = DDSCR(in_channels, 12)
        self.enc2_flair = DDSCR(12, 24)
        self.enc3_flair = DDSCR(24, 48)
        self.enc4_flair = DDSCR(48, 96)

        # Bottleneck blocks (96*4 = 384 input channels)
        self.bottleneck = LBB(384)

        # Decoder blocks
        self.dec1 = DDSCR(768, 96)    # 384 + 384 = 768

        self.upconv1 = nn.ConvTranspose3d(96, 48, kernel_size=2, stride=2)
        self.dec2 = DDSCR(240, 48)    # 48 + 192 = 240 (48*4)

        self.upconv2 = nn.ConvTranspose3d(48, 24, kernel_size=2, stride=2)
        self.dec3 = DDSCR(120, 24)    # 24 + 96 = 120 (24*4)

        self.upconv3 = nn.ConvTranspose3d(24, 12, kernel_size=2, stride=2)
        self.dec4 = DDSCR(60, 12)     # 12 + 48 = 60 (12*4)

        self.final = nn.Conv3d(12, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):
        # Encoder Path - T1
        e1_t1 = self.enc1_t1(t1)
        d1_t1 = self.pool(e1_t1)
        e2_t1 = self.enc2_t1(d1_t1)
        d2_t1 = self.pool(e2_t1)
        e3_t1 = self.enc3_t1(d2_t1)
        d3_t1 = self.pool(e3_t1)
        e4_t1 = self.enc4_t1(d3_t1)

        # Encoder Path - T1CE
        e1_t1ce = self.enc1_t1ce(t1ce)
        d1_t1ce = self.pool(e1_t1ce)
        e2_t1ce = self.enc2_t1ce(d1_t1ce)
        d2_t1ce = self.pool(e2_t1ce)
        e3_t1ce = self.enc3_t1ce(d2_t1ce)
        d3_t1ce = self.pool(e3_t1ce)
        e4_t1ce = self.enc4_t1ce(d3_t1ce)

        # Encoder Path - T2
        e1_t2 = self.enc1_t2(t2)
        d1_t2 = self.pool(e1_t2)
        e2_t2 = self.enc2_t2(d1_t2)
        d2_t2 = self.pool(e2_t2)
        e3_t2 = self.enc3_t2(d2_t2)
        d3_t2 = self.pool(e3_t2)
        e4_t2 = self.enc4_t2(d3_t2)

        # Encoder Path - FLAIR
        e1_flair = self.enc1_flair(flair)
        d1_flair = self.pool(e1_flair)
        e2_flair = self.enc2_flair(d1_flair)
        d2_flair = self.pool(e2_flair)
        e3_flair = self.enc3_flair(d2_flair)
        d3_flair = self.pool(e3_flair)
        e4_flair = self.enc4_flair(d3_flair)

        # Merge encoder outputs
        merged_e4 = torch.cat([e4_t1, e4_t1ce, e4_t2, e4_flair], dim=1)      # 96*4 = 384
        merged_e3 = torch.cat([e3_t1, e3_t1ce, e3_t2, e3_flair], dim=1)      # 48*4 = 192
        merged_e2 = torch.cat([e2_t1, e2_t1ce, e2_t2, e2_flair], dim=1)      # 24*4 = 96
        merged_e1 = torch.cat([e1_t1, e1_t1ce, e1_t2, e1_flair], dim=1)      # 12*4 = 48

        # Bottleneck
        bottleneck = self.bottleneck(merged_e4)

        # Decoder Path
        merge5 = torch.cat([bottleneck, merged_e4], dim=1)
        c5 = self.dec1(merge5)

        up6 = self.upconv1(c5)
        merge6 = torch.cat([up6, merged_e3], dim=1)
        c6 = self.dec2(merge6)

        up7 = self.upconv2(c6)
        merge7 = torch.cat([up7, merged_e2], dim=1)
        c7 = self.dec3(merge7)

        up8 = self.upconv3(c7)
        merge8 = torch.cat([up8, merged_e1], dim=1)
        c8 = self.dec4(merge8)

        out = self.final(c8)
        return out


class base_DME(nn.Module):
    """
    Multi-Modal Brain Tumor Segmentation with Dual-Stream Fusion:
    - DME (Differential Modality Encoding) at skip connections
    - LBB at bottleneck for global feature integration

    Architecture Philosophy:
    - Skip connections: Preserve discriminative boundaries via differences
    - Bottleneck: Integrate semantic content via concatenation
    """

    def __init__(self, img_h, img_w, img_d, in_channels=1, num_classes=4):
        super().__init__()

        # Max pooling layer (shared across all branches)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder blocks for T1
        self.enc1_t1 = DDSCR(in_channels, 12)
        self.enc2_t1 = DDSCR(12, 24)
        self.enc3_t1 = DDSCR(24, 48)
        self.enc4_t1 = DDSCR(48, 96)

        # Encoder blocks for T1CE
        self.enc1_t1ce = DDSCR(in_channels, 12)
        self.enc2_t1ce = DDSCR(12, 24)
        self.enc3_t1ce = DDSCR(24, 48)
        self.enc4_t1ce = DDSCR(48, 96)

        # Encoder blocks for T2
        self.enc1_t2 = DDSCR(in_channels, 12)
        self.enc2_t2 = DDSCR(12, 24)
        self.enc3_t2 = DDSCR(24, 48)
        self.enc4_t2 = DDSCR(48, 96)

        # Encoder blocks for FLAIR
        self.enc1_flair = DDSCR(in_channels, 12)
        self.enc2_flair = DDSCR(12, 24)
        self.enc3_flair = DDSCR(24, 48)
        self.enc4_flair = DDSCR(48, 96)

        # Each DME takes 4 modalities and outputs 2x channels for skip connections
        self.dme_skip1 = DME(12, out_channels=24)
        self.dme_skip2 = DME(24, out_channels=48)
        self.dme_skip3 = DME(48, out_channels=96)
        self.dme_skip4 = DME(96, out_channels=192)

        self.bottleneck = LBB(384)  # 96*4 = 384 input channels

        self.dec1 = DDSCR(576, 96)  # 384 (bottleneck) + 192 (dme_skip4) = 576

        self.upconv1 = nn.ConvTranspose3d(96, 48, kernel_size=2, stride=2)
        self.dec2 = DDSCR(144, 48)  # 48 (upconv1) + 96 (dme_skip3) = 144

        self.upconv2 = nn.ConvTranspose3d(48, 24, kernel_size=2, stride=2)
        self.dec3 = DDSCR(72, 24)  # 24 (upconv2) + 48 (dme_skip2) = 72

        self.upconv3 = nn.ConvTranspose3d(24, 12, kernel_size=2, stride=2)
        self.dec4 = DDSCR(36, 12)  # 12 (upconv3) + 24 (dme_skip1) = 36

        self.final = nn.Conv3d(12, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):
        """
        Forward pass with dual-stream fusion

        Args:
            t1, t1ce, t2, flair: [B, 1, D, H, W] input modalities

        Returns:
            out: [B, num_classes, D, H, W] segmentation logits
        """
        # T1 pathway
        e1_t1 = self.enc1_t1(t1)
        d1_t1 = self.pool(e1_t1)
        e2_t1 = self.enc2_t1(d1_t1)
        d2_t1 = self.pool(e2_t1)
        e3_t1 = self.enc3_t1(d2_t1)
        d3_t1 = self.pool(e3_t1)
        e4_t1 = self.enc4_t1(d3_t1)

        # T1CE pathway
        e1_t1ce = self.enc1_t1ce(t1ce)
        d1_t1ce = self.pool(e1_t1ce)
        e2_t1ce = self.enc2_t1ce(d1_t1ce)
        d2_t1ce = self.pool(e2_t1ce)
        e3_t1ce = self.enc3_t1ce(d2_t1ce)
        d3_t1ce = self.pool(e3_t1ce)
        e4_t1ce = self.enc4_t1ce(d3_t1ce)

        # T2 pathway
        e1_t2 = self.enc1_t2(t2)
        d1_t2 = self.pool(e1_t2)
        e2_t2 = self.enc2_t2(d1_t2)
        d2_t2 = self.pool(e2_t2)
        e3_t2 = self.enc3_t2(d2_t2)
        d3_t2 = self.pool(e3_t2)
        e4_t2 = self.enc4_t2(d3_t2)

        # FLAIR pathway
        e1_flair = self.enc1_flair(flair)
        d1_flair = self.pool(e1_flair)
        e2_flair = self.enc2_flair(d1_flair)
        d2_flair = self.pool(e2_flair)
        e3_flair = self.enc3_flair(d2_flair)
        d3_flair = self.pool(e3_flair)
        e4_flair = self.enc4_flair(d3_flair)

        dme_e1 = self.dme_skip1(e1_t1, e1_t1ce, e1_t2, e1_flair)
        dme_e2 = self.dme_skip2(e2_t1, e2_t1ce, e2_t2, e2_flair)
        dme_e3 = self.dme_skip3(e3_t1, e3_t1ce, e3_t2, e3_flair)
        dme_e4 = self.dme_skip4(e4_t1, e4_t1ce, e4_t2, e4_flair)

        merged_e4 = torch.cat([e4_t1, e4_t1ce, e4_t2, e4_flair], dim=1)
        bottleneck = self.bottleneck(merged_e4)

        merge5 = torch.cat([bottleneck, dme_e4], dim=1)
        c5 = self.dec1(merge5)

        up6 = self.upconv1(c5)
        merge6 = torch.cat([up6, dme_e3], dim=1)
        c6 = self.dec2(merge6)

        up7 = self.upconv2(c6)
        merge7 = torch.cat([up7, dme_e2], dim=1)
        c7 = self.dec3(merge7)

        up8 = self.upconv3(c7)
        merge8 = torch.cat([up8, dme_e1], dim=1)
        c8 = self.dec4(merge8)

        out = self.final(c8)
        return out

class base_DME1(nn.Module):
    """
    Multi-Modal Brain Tumor Segmentation with Dual-Stream Fusion:
    - DME (Differential Modality Encoding) at skip connections
    - LBB at bottleneck for global feature integration

    Architecture Philosophy:
    - Skip connections: Preserve discriminative boundaries via differences
    - Bottleneck: Integrate semantic content via concatenation
    """

    def __init__(self, img_h, img_w, img_d, in_channels=1, num_classes=4):
        super().__init__()

        # Max pooling layer (shared across all branches)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder blocks for T1
        self.enc1_t1 = DDSCR(in_channels, 12)
        self.enc2_t1 = DDSCR(12, 24)
        self.enc3_t1 = DDSCR(24, 48)
        self.enc4_t1 = DDSCR(48, 96)

        # Encoder blocks for T1CE
        self.enc1_t1ce = DDSCR(in_channels, 12)
        self.enc2_t1ce = DDSCR(12, 24)
        self.enc3_t1ce = DDSCR(24, 48)
        self.enc4_t1ce = DDSCR(48, 96)

        # Encoder blocks for T2
        self.enc1_t2 = DDSCR(in_channels, 12)
        self.enc2_t2 = DDSCR(12, 24)
        self.enc3_t2 = DDSCR(24, 48)
        self.enc4_t2 = DDSCR(48, 96)

        # Encoder blocks for FLAIR
        self.enc1_flair = DDSCR(in_channels, 12)
        self.enc2_flair = DDSCR(12, 24)
        self.enc3_flair = DDSCR(24, 48)
        self.enc4_flair = DDSCR(48, 96)

        # Each DME takes 4 modalities and outputs 2x channels for skip connections
        self.dme_skip1 = DME1(12, out_channels=24)
        self.dme_skip2 = DME1(24, out_channels=48)
        self.dme_skip3 = DME1(48, out_channels=96)
        self.dme_skip4 = DME1(96, out_channels=192)

        self.bottleneck = LBB(384)  # 96*4 = 384 input channels

        self.dec1 = DDSCR(576, 96)  # 384 (bottleneck) + 192 (dme_skip4) = 576

        self.upconv1 = nn.ConvTranspose3d(96, 48, kernel_size=2, stride=2)
        self.dec2 = DDSCR(144, 48)  # 48 (upconv1) + 96 (dme_skip3) = 144

        self.upconv2 = nn.ConvTranspose3d(48, 24, kernel_size=2, stride=2)
        self.dec3 = DDSCR(72, 24)  # 24 (upconv2) + 48 (dme_skip2) = 72

        self.upconv3 = nn.ConvTranspose3d(24, 12, kernel_size=2, stride=2)
        self.dec4 = DDSCR(36, 12)  # 12 (upconv3) + 24 (dme_skip1) = 36

        self.final = nn.Conv3d(12, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):
        """
        Forward pass with dual-stream fusion

        Args:
            t1, t1ce, t2, flair: [B, 1, D, H, W] input modalities

        Returns:
            out: [B, num_classes, D, H, W] segmentation logits
        """
        # T1 pathway
        e1_t1 = self.enc1_t1(t1)
        d1_t1 = self.pool(e1_t1)
        e2_t1 = self.enc2_t1(d1_t1)
        d2_t1 = self.pool(e2_t1)
        e3_t1 = self.enc3_t1(d2_t1)
        d3_t1 = self.pool(e3_t1)
        e4_t1 = self.enc4_t1(d3_t1)

        # T1CE pathway
        e1_t1ce = self.enc1_t1ce(t1ce)
        d1_t1ce = self.pool(e1_t1ce)
        e2_t1ce = self.enc2_t1ce(d1_t1ce)
        d2_t1ce = self.pool(e2_t1ce)
        e3_t1ce = self.enc3_t1ce(d2_t1ce)
        d3_t1ce = self.pool(e3_t1ce)
        e4_t1ce = self.enc4_t1ce(d3_t1ce)

        # T2 pathway
        e1_t2 = self.enc1_t2(t2)
        d1_t2 = self.pool(e1_t2)
        e2_t2 = self.enc2_t2(d1_t2)
        d2_t2 = self.pool(e2_t2)
        e3_t2 = self.enc3_t2(d2_t2)
        d3_t2 = self.pool(e3_t2)
        e4_t2 = self.enc4_t2(d3_t2)

        # FLAIR pathway
        e1_flair = self.enc1_flair(flair)
        d1_flair = self.pool(e1_flair)
        e2_flair = self.enc2_flair(d1_flair)
        d2_flair = self.pool(e2_flair)
        e3_flair = self.enc3_flair(d2_flair)
        d3_flair = self.pool(e3_flair)
        e4_flair = self.enc4_flair(d3_flair)

        dme_e1 = self.dme_skip1(e1_t1, e1_t1ce, e1_t2, e1_flair)
        dme_e2 = self.dme_skip2(e2_t1, e2_t1ce, e2_t2, e2_flair)
        dme_e3 = self.dme_skip3(e3_t1, e3_t1ce, e3_t2, e3_flair)
        dme_e4 = self.dme_skip4(e4_t1, e4_t1ce, e4_t2, e4_flair)

        merged_e4 = torch.cat([e4_t1, e4_t1ce, e4_t2, e4_flair], dim=1)
        bottleneck = self.bottleneck(merged_e4)

        merge5 = torch.cat([bottleneck, dme_e4], dim=1)
        c5 = self.dec1(merge5)

        up6 = self.upconv1(c5)
        merge6 = torch.cat([up6, dme_e3], dim=1)
        c6 = self.dec2(merge6)

        up7 = self.upconv2(c6)
        merge7 = torch.cat([up7, dme_e2], dim=1)
        c7 = self.dec3(merge7)

        up8 = self.upconv3(c7)
        merge8 = torch.cat([up8, dme_e1], dim=1)
        c8 = self.dec4(merge8)

        out = self.final(c8)
        return out

class base_DME2(nn.Module):
    """
    Multi-Modal Brain Tumor Segmentation with Dual-Stream Fusion:
    - DME (Differential Modality Encoding) at skip connections
    - LBB at bottleneck for global feature integration

    Architecture Philosophy:
    - Skip connections: Preserve discriminative boundaries via differences
    - Bottleneck: Integrate semantic content via concatenation
    """

    def __init__(self, img_h, img_w, img_d, in_channels=1, num_classes=4):
        super().__init__()

        # Max pooling layer (shared across all branches)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder blocks for T1
        self.enc1_t1 = DDSCR(in_channels, 12)
        self.enc2_t1 = DDSCR(12, 24)
        self.enc3_t1 = DDSCR(24, 48)
        self.enc4_t1 = DDSCR(48, 96)

        # Encoder blocks for T1CE
        self.enc1_t1ce = DDSCR(in_channels, 12)
        self.enc2_t1ce = DDSCR(12, 24)
        self.enc3_t1ce = DDSCR(24, 48)
        self.enc4_t1ce = DDSCR(48, 96)

        # Encoder blocks for T2
        self.enc1_t2 = DDSCR(in_channels, 12)
        self.enc2_t2 = DDSCR(12, 24)
        self.enc3_t2 = DDSCR(24, 48)
        self.enc4_t2 = DDSCR(48, 96)

        # Encoder blocks for FLAIR
        self.enc1_flair = DDSCR(in_channels, 12)
        self.enc2_flair = DDSCR(12, 24)
        self.enc3_flair = DDSCR(24, 48)
        self.enc4_flair = DDSCR(48, 96)

        # Each DME takes 4 modalities and outputs 2x channels for skip connections
        self.dme_skip1 = DME2(12, out_channels=24)
        self.dme_skip2 = DME2(24, out_channels=48)
        self.dme_skip3 = DME2(48, out_channels=96)
        self.dme_skip4 = DME2(96, out_channels=192)

        self.bottleneck = LBB(384)  # 96*4 = 384 input channels

        self.dec1 = DDSCR(576, 96)  # 384 (bottleneck) + 192 (dme_skip4) = 576

        self.upconv1 = nn.ConvTranspose3d(96, 48, kernel_size=2, stride=2)
        self.dec2 = DDSCR(144, 48)  # 48 (upconv1) + 96 (dme_skip3) = 144

        self.upconv2 = nn.ConvTranspose3d(48, 24, kernel_size=2, stride=2)
        self.dec3 = DDSCR(72, 24)  # 24 (upconv2) + 48 (dme_skip2) = 72

        self.upconv3 = nn.ConvTranspose3d(24, 12, kernel_size=2, stride=2)
        self.dec4 = DDSCR(36, 12)  # 12 (upconv3) + 24 (dme_skip1) = 36

        self.final = nn.Conv3d(12, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):
        """
        Forward pass with dual-stream fusion

        Args:
            t1, t1ce, t2, flair: [B, 1, D, H, W] input modalities

        Returns:
            out: [B, num_classes, D, H, W] segmentation logits
        """
        # T1 pathway
        e1_t1 = self.enc1_t1(t1)
        d1_t1 = self.pool(e1_t1)
        e2_t1 = self.enc2_t1(d1_t1)
        d2_t1 = self.pool(e2_t1)
        e3_t1 = self.enc3_t1(d2_t1)
        d3_t1 = self.pool(e3_t1)
        e4_t1 = self.enc4_t1(d3_t1)

        # T1CE pathway
        e1_t1ce = self.enc1_t1ce(t1ce)
        d1_t1ce = self.pool(e1_t1ce)
        e2_t1ce = self.enc2_t1ce(d1_t1ce)
        d2_t1ce = self.pool(e2_t1ce)
        e3_t1ce = self.enc3_t1ce(d2_t1ce)
        d3_t1ce = self.pool(e3_t1ce)
        e4_t1ce = self.enc4_t1ce(d3_t1ce)

        # T2 pathway
        e1_t2 = self.enc1_t2(t2)
        d1_t2 = self.pool(e1_t2)
        e2_t2 = self.enc2_t2(d1_t2)
        d2_t2 = self.pool(e2_t2)
        e3_t2 = self.enc3_t2(d2_t2)
        d3_t2 = self.pool(e3_t2)
        e4_t2 = self.enc4_t2(d3_t2)

        # FLAIR pathway
        e1_flair = self.enc1_flair(flair)
        d1_flair = self.pool(e1_flair)
        e2_flair = self.enc2_flair(d1_flair)
        d2_flair = self.pool(e2_flair)
        e3_flair = self.enc3_flair(d2_flair)
        d3_flair = self.pool(e3_flair)
        e4_flair = self.enc4_flair(d3_flair)

        dme_e1 = self.dme_skip1(e1_t1, e1_t1ce, e1_t2, e1_flair)
        dme_e2 = self.dme_skip2(e2_t1, e2_t1ce, e2_t2, e2_flair)
        dme_e3 = self.dme_skip3(e3_t1, e3_t1ce, e3_t2, e3_flair)
        dme_e4 = self.dme_skip4(e4_t1, e4_t1ce, e4_t2, e4_flair)

        merged_e4 = torch.cat([e4_t1, e4_t1ce, e4_t2, e4_flair], dim=1)
        bottleneck = self.bottleneck(merged_e4)

        merge5 = torch.cat([bottleneck, dme_e4], dim=1)
        c5 = self.dec1(merge5)

        up6 = self.upconv1(c5)
        merge6 = torch.cat([up6, dme_e3], dim=1)
        c6 = self.dec2(merge6)

        up7 = self.upconv2(c6)
        merge7 = torch.cat([up7, dme_e2], dim=1)
        c7 = self.dec3(merge7)

        up8 = self.upconv3(c7)
        merge8 = torch.cat([up8, dme_e1], dim=1)
        c8 = self.dec4(merge8)

        out = self.final(c8)
        return out

class base_DME3(nn.Module):
    """
    Multi-Modal Brain Tumor Segmentation with Dual-Stream Fusion:
    - DME (Differential Modality Encoding) at skip connections
    - LBB at bottleneck for global feature integration

    Architecture Philosophy:
    - Skip connections: Preserve discriminative boundaries via differences
    - Bottleneck: Integrate semantic content via concatenation
    """

    def __init__(self, img_h, img_w, img_d, in_channels=1, num_classes=4):
        super().__init__()

        # Max pooling layer (shared across all branches)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder blocks for T1
        self.enc1_t1 = DDSCR(in_channels, 12)
        self.enc2_t1 = DDSCR(12, 24)
        self.enc3_t1 = DDSCR(24, 48)
        self.enc4_t1 = DDSCR(48, 96)

        # Encoder blocks for T1CE
        self.enc1_t1ce = DDSCR(in_channels, 12)
        self.enc2_t1ce = DDSCR(12, 24)
        self.enc3_t1ce = DDSCR(24, 48)
        self.enc4_t1ce = DDSCR(48, 96)

        # Encoder blocks for T2
        self.enc1_t2 = DDSCR(in_channels, 12)
        self.enc2_t2 = DDSCR(12, 24)
        self.enc3_t2 = DDSCR(24, 48)
        self.enc4_t2 = DDSCR(48, 96)

        # Encoder blocks for FLAIR
        self.enc1_flair = DDSCR(in_channels, 12)
        self.enc2_flair = DDSCR(12, 24)
        self.enc3_flair = DDSCR(24, 48)
        self.enc4_flair = DDSCR(48, 96)

        # Each DME takes 4 modalities and outputs 2x channels for skip connections
        self.dme_skip1 = DME3(12, out_channels=24)
        self.dme_skip2 = DME3(24, out_channels=48)
        self.dme_skip3 = DME3(48, out_channels=96)
        self.dme_skip4 = DME3(96, out_channels=192)

        self.bottleneck = LBB(384)  # 96*4 = 384 input channels

        self.dec1 = DDSCR(576, 96)  # 384 (bottleneck) + 192 (dme_skip4) = 576

        self.upconv1 = nn.ConvTranspose3d(96, 48, kernel_size=2, stride=2)
        self.dec2 = DDSCR(144, 48)  # 48 (upconv1) + 96 (dme_skip3) = 144

        self.upconv2 = nn.ConvTranspose3d(48, 24, kernel_size=2, stride=2)
        self.dec3 = DDSCR(72, 24)  # 24 (upconv2) + 48 (dme_skip2) = 72

        self.upconv3 = nn.ConvTranspose3d(24, 12, kernel_size=2, stride=2)
        self.dec4 = DDSCR(36, 12)  # 12 (upconv3) + 24 (dme_skip1) = 36

        self.final = nn.Conv3d(12, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):
        """
        Forward pass with dual-stream fusion

        Args:
            t1, t1ce, t2, flair: [B, 1, D, H, W] input modalities

        Returns:
            out: [B, num_classes, D, H, W] segmentation logits
        """
        # T1 pathway
        e1_t1 = self.enc1_t1(t1)
        d1_t1 = self.pool(e1_t1)
        e2_t1 = self.enc2_t1(d1_t1)
        d2_t1 = self.pool(e2_t1)
        e3_t1 = self.enc3_t1(d2_t1)
        d3_t1 = self.pool(e3_t1)
        e4_t1 = self.enc4_t1(d3_t1)

        # T1CE pathway
        e1_t1ce = self.enc1_t1ce(t1ce)
        d1_t1ce = self.pool(e1_t1ce)
        e2_t1ce = self.enc2_t1ce(d1_t1ce)
        d2_t1ce = self.pool(e2_t1ce)
        e3_t1ce = self.enc3_t1ce(d2_t1ce)
        d3_t1ce = self.pool(e3_t1ce)
        e4_t1ce = self.enc4_t1ce(d3_t1ce)

        # T2 pathway
        e1_t2 = self.enc1_t2(t2)
        d1_t2 = self.pool(e1_t2)
        e2_t2 = self.enc2_t2(d1_t2)
        d2_t2 = self.pool(e2_t2)
        e3_t2 = self.enc3_t2(d2_t2)
        d3_t2 = self.pool(e3_t2)
        e4_t2 = self.enc4_t2(d3_t2)

        # FLAIR pathway
        e1_flair = self.enc1_flair(flair)
        d1_flair = self.pool(e1_flair)
        e2_flair = self.enc2_flair(d1_flair)
        d2_flair = self.pool(e2_flair)
        e3_flair = self.enc3_flair(d2_flair)
        d3_flair = self.pool(e3_flair)
        e4_flair = self.enc4_flair(d3_flair)

        dme_e1 = self.dme_skip1(e1_t1, e1_t1ce, e1_t2, e1_flair)
        dme_e2 = self.dme_skip2(e2_t1, e2_t1ce, e2_t2, e2_flair)
        dme_e3 = self.dme_skip3(e3_t1, e3_t1ce, e3_t2, e3_flair)
        dme_e4 = self.dme_skip4(e4_t1, e4_t1ce, e4_t2, e4_flair)

        merged_e4 = torch.cat([e4_t1, e4_t1ce, e4_t2, e4_flair], dim=1)
        bottleneck = self.bottleneck(merged_e4)

        merge5 = torch.cat([bottleneck, dme_e4], dim=1)
        c5 = self.dec1(merge5)

        up6 = self.upconv1(c5)
        merge6 = torch.cat([up6, dme_e3], dim=1)
        c6 = self.dec2(merge6)

        up7 = self.upconv2(c6)
        merge7 = torch.cat([up7, dme_e2], dim=1)
        c7 = self.dec3(merge7)

        up8 = self.upconv3(c7)
        merge8 = torch.cat([up8, dme_e1], dim=1)
        c8 = self.dec4(merge8)

        out = self.final(c8)
        return out


class base_PGAM(nn.Module):
    def __init__(self, img_h, img_w, img_d, in_channels=1, num_classes=4):
        super().__init__()

        # Store original input size
        self.img_d = img_d
        self.img_h = img_h
        self.img_w = img_w

        # Max pooling layer
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder blocks for T1
        self.enc1_t1 = DDSCR(in_channels, 12)
        self.enc2_t1 = DDSCR(12, 24)
        self.enc3_t1 = DDSCR(24, 48)
        self.enc4_t1 = DDSCR(48, 96)

        # Encoder blocks for T1CE
        self.enc1_t1ce = DDSCR(in_channels, 12)
        self.enc2_t1ce = DDSCR(12, 24)
        self.enc3_t1ce = DDSCR(24, 48)
        self.enc4_t1ce = DDSCR(48, 96)

        # Encoder blocks for T2
        self.enc1_t2 = DDSCR(in_channels, 12)
        self.enc2_t2 = DDSCR(12, 24)
        self.enc3_t2 = DDSCR(24, 48)
        self.enc4_t2 = DDSCR(48, 96)

        # Encoder blocks for FLAIR
        self.enc1_flair = DDSCR(in_channels, 12)
        self.enc2_flair = DDSCR(12, 24)
        self.enc3_flair = DDSCR(24, 48)
        self.enc4_flair = DDSCR(48, 96)

        # Bottleneck (96*4 = 384 input channels)
        self.bottleneck = LBB(384)

        # Decoder blocks
        self.dec1 = DDSCR(768, 96)     # 384 + 384 = 768
        self.upconv1 = nn.ConvTranspose3d(96, 48, kernel_size=2, stride=2)

        self.dec2 = DDSCR(240, 48)     # 48 + 192 = 240 (48*4)
        self.upconv2 = nn.ConvTranspose3d(48, 24, kernel_size=2, stride=2)

        self.dec3 = DDSCR(120, 24)     # 24 + 96 = 120 (24*4)
        self.upconv3 = nn.ConvTranspose3d(24, 12, kernel_size=2, stride=2)

        self.dec4 = DDSCR(60, 12)      # 12 + 48 = 60 (12*4)

        self.fusion = PGAM(
            feature_channels=[12, 24, 48, 96],  # From finest to coarsest
            output_size=(img_d, img_h, img_w),
            num_classes=num_classes,
            intermediate_channels=12
        )

    def forward(self, t1, t1ce, t2, flair):
        # Encoder Path - T1
        e1_t1 = self.enc1_t1(t1)
        d1_t1 = self.pool(e1_t1)
        e2_t1 = self.enc2_t1(d1_t1)
        d2_t1 = self.pool(e2_t1)
        e3_t1 = self.enc3_t1(d2_t1)
        d3_t1 = self.pool(e3_t1)
        e4_t1 = self.enc4_t1(d3_t1)

        # Encoder Path - T1CE
        e1_t1ce = self.enc1_t1ce(t1ce)
        d1_t1ce = self.pool(e1_t1ce)
        e2_t1ce = self.enc2_t1ce(d1_t1ce)
        d2_t1ce = self.pool(e2_t1ce)
        e3_t1ce = self.enc3_t1ce(d2_t1ce)
        d3_t1ce = self.pool(e3_t1ce)
        e4_t1ce = self.enc4_t1ce(d3_t1ce)

        # Encoder Path - T2
        e1_t2 = self.enc1_t2(t2)
        d1_t2 = self.pool(e1_t2)
        e2_t2 = self.enc2_t2(d1_t2)
        d2_t2 = self.pool(e2_t2)
        e3_t2 = self.enc3_t2(d2_t2)
        d3_t2 = self.pool(e3_t2)
        e4_t2 = self.enc4_t2(d3_t2)

        # Encoder Path - FLAIR
        e1_flair = self.enc1_flair(flair)
        d1_flair = self.pool(e1_flair)
        e2_flair = self.enc2_flair(d1_flair)
        d2_flair = self.pool(e2_flair)
        e3_flair = self.enc3_flair(d2_flair)
        d3_flair = self.pool(e3_flair)
        e4_flair = self.enc4_flair(d3_flair)

        # Merge encoder outputs
        merged_e4 = torch.cat([e4_t1, e4_t1ce, e4_t2, e4_flair], dim=1)      # 96*4 = 384
        merged_e3 = torch.cat([e3_t1, e3_t1ce, e3_t2, e3_flair], dim=1)      # 48*4 = 192
        merged_e2 = torch.cat([e2_t1, e2_t1ce, e2_t2, e2_flair], dim=1)      # 24*4 = 96
        merged_e1 = torch.cat([e1_t1, e1_t1ce, e1_t2, e1_flair], dim=1)      # 12*4 = 48

        # Bottleneck
        bottleneck = self.bottleneck(merged_e4)

        # Decoder Path
        merge5 = torch.cat([bottleneck, merged_e4], dim=1)
        c5 = self.dec1(merge5)

        up6 = self.upconv1(c5)
        merge6 = torch.cat([up6, merged_e3], dim=1)
        c6 = self.dec2(merge6)

        up7 = self.upconv2(c6)
        merge7 = torch.cat([up7, merged_e2], dim=1)
        c7 = self.dec3(merge7)

        up8 = self.upconv3(c7)
        merge8 = torch.cat([up8, merged_e1], dim=1)
        c8 = self.dec4(merge8)

        # Multi-Scale Feature Fusion
        output = self.fusion([c8, c7, c6, c5])

        return output

class base_PGAM1(nn.Module):
    def __init__(self, img_h, img_w, img_d, in_channels=1, num_classes=4):
        super().__init__()

        # Store original input size
        self.img_d = img_d
        self.img_h = img_h
        self.img_w = img_w

        # Max pooling layer
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder blocks for T1
        self.enc1_t1 = DDSCR(in_channels, 12)
        self.enc2_t1 = DDSCR(12, 24)
        self.enc3_t1 = DDSCR(24, 48)
        self.enc4_t1 = DDSCR(48, 96)

        # Encoder blocks for T1CE
        self.enc1_t1ce = DDSCR(in_channels, 12)
        self.enc2_t1ce = DDSCR(12, 24)
        self.enc3_t1ce = DDSCR(24, 48)
        self.enc4_t1ce = DDSCR(48, 96)

        # Encoder blocks for T2
        self.enc1_t2 = DDSCR(in_channels, 12)
        self.enc2_t2 = DDSCR(12, 24)
        self.enc3_t2 = DDSCR(24, 48)
        self.enc4_t2 = DDSCR(48, 96)

        # Encoder blocks for FLAIR
        self.enc1_flair = DDSCR(in_channels, 12)
        self.enc2_flair = DDSCR(12, 24)
        self.enc3_flair = DDSCR(24, 48)
        self.enc4_flair = DDSCR(48, 96)

        # Bottleneck (96*4 = 384 input channels)
        self.bottleneck = LBB(384)

        # Decoder blocks
        self.dec1 = DDSCR(768, 96)     # 384 + 384 = 768
        self.upconv1 = nn.ConvTranspose3d(96, 48, kernel_size=2, stride=2)

        self.dec2 = DDSCR(240, 48)     # 48 + 192 = 240 (48*4)
        self.upconv2 = nn.ConvTranspose3d(48, 24, kernel_size=2, stride=2)

        self.dec3 = DDSCR(120, 24)     # 24 + 96 = 120 (24*4)
        self.upconv3 = nn.ConvTranspose3d(24, 12, kernel_size=2, stride=2)

        self.dec4 = DDSCR(60, 12)      # 12 + 48 = 60 (12*4)

        self.fusion = PGAM1(
            feature_channels=[12, 24, 48, 96],
            output_size=(img_d, img_h, img_w),
            num_classes=num_classes,
            intermediate_channels=12
        )

    def forward(self, t1, t1ce, t2, flair):
        # Encoder Path - T1
        e1_t1 = self.enc1_t1(t1)
        d1_t1 = self.pool(e1_t1)
        e2_t1 = self.enc2_t1(d1_t1)
        d2_t1 = self.pool(e2_t1)
        e3_t1 = self.enc3_t1(d2_t1)
        d3_t1 = self.pool(e3_t1)
        e4_t1 = self.enc4_t1(d3_t1)

        # Encoder Path - T1CE
        e1_t1ce = self.enc1_t1ce(t1ce)
        d1_t1ce = self.pool(e1_t1ce)
        e2_t1ce = self.enc2_t1ce(d1_t1ce)
        d2_t1ce = self.pool(e2_t1ce)
        e3_t1ce = self.enc3_t1ce(d2_t1ce)
        d3_t1ce = self.pool(e3_t1ce)
        e4_t1ce = self.enc4_t1ce(d3_t1ce)

        # Encoder Path - T2
        e1_t2 = self.enc1_t2(t2)
        d1_t2 = self.pool(e1_t2)
        e2_t2 = self.enc2_t2(d1_t2)
        d2_t2 = self.pool(e2_t2)
        e3_t2 = self.enc3_t2(d2_t2)
        d3_t2 = self.pool(e3_t2)
        e4_t2 = self.enc4_t2(d3_t2)

        # Encoder Path - FLAIR
        e1_flair = self.enc1_flair(flair)
        d1_flair = self.pool(e1_flair)
        e2_flair = self.enc2_flair(d1_flair)
        d2_flair = self.pool(e2_flair)
        e3_flair = self.enc3_flair(d2_flair)
        d3_flair = self.pool(e3_flair)
        e4_flair = self.enc4_flair(d3_flair)

        # Merge encoder outputs
        merged_e4 = torch.cat([e4_t1, e4_t1ce, e4_t2, e4_flair], dim=1)      # 96*4 = 384
        merged_e3 = torch.cat([e3_t1, e3_t1ce, e3_t2, e3_flair], dim=1)      # 48*4 = 192
        merged_e2 = torch.cat([e2_t1, e2_t1ce, e2_t2, e2_flair], dim=1)      # 24*4 = 96
        merged_e1 = torch.cat([e1_t1, e1_t1ce, e1_t2, e1_flair], dim=1)      # 12*4 = 48

        # Bottleneck
        bottleneck = self.bottleneck(merged_e4)

        # Decoder Path
        merge5 = torch.cat([bottleneck, merged_e4], dim=1)
        c5 = self.dec1(merge5)

        up6 = self.upconv1(c5)
        merge6 = torch.cat([up6, merged_e3], dim=1)
        c6 = self.dec2(merge6)

        up7 = self.upconv2(c6)
        merge7 = torch.cat([up7, merged_e2], dim=1)
        c7 = self.dec3(merge7)

        up8 = self.upconv3(c7)
        merge8 = torch.cat([up8, merged_e1], dim=1)
        c8 = self.dec4(merge8)

        # Multi-Scale Feature Fusion
        output = self.fusion([c8, c7, c6, c5])
        return output

class base_PGAM2(nn.Module):
    def __init__(self, img_h, img_w, img_d, in_channels=1, num_classes=4):
        super().__init__()

        # Store original input size
        self.img_d = img_d
        self.img_h = img_h
        self.img_w = img_w

        # Max pooling layer
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder blocks for T1
        self.enc1_t1 = DDSCR(in_channels, 12)
        self.enc2_t1 = DDSCR(12, 24)
        self.enc3_t1 = DDSCR(24, 48)
        self.enc4_t1 = DDSCR(48, 96)

        # Encoder blocks for T1CE
        self.enc1_t1ce = DDSCR(in_channels, 12)
        self.enc2_t1ce = DDSCR(12, 24)
        self.enc3_t1ce = DDSCR(24, 48)
        self.enc4_t1ce = DDSCR(48, 96)

        # Encoder blocks for T2
        self.enc1_t2 = DDSCR(in_channels, 12)
        self.enc2_t2 = DDSCR(12, 24)
        self.enc3_t2 = DDSCR(24, 48)
        self.enc4_t2 = DDSCR(48, 96)

        # Encoder blocks for FLAIR
        self.enc1_flair = DDSCR(in_channels, 12)
        self.enc2_flair = DDSCR(12, 24)
        self.enc3_flair = DDSCR(24, 48)
        self.enc4_flair = DDSCR(48, 96)

        # Bottleneck (96*4 = 384 input channels)
        self.bottleneck = LBB(384)

        # Decoder blocks
        self.dec1 = DDSCR(768, 96)     # 384 + 384 = 768
        self.upconv1 = nn.ConvTranspose3d(96, 48, kernel_size=2, stride=2)

        self.dec2 = DDSCR(240, 48)     # 48 + 192 = 240 (48*4)
        self.upconv2 = nn.ConvTranspose3d(48, 24, kernel_size=2, stride=2)

        self.dec3 = DDSCR(120, 24)     # 24 + 96 = 120 (24*4)
        self.upconv3 = nn.ConvTranspose3d(24, 12, kernel_size=2, stride=2)

        self.dec4 = DDSCR(60, 12)      # 12 + 48 = 60 (12*4)

        self.fusion = PGAM2(
            feature_channels=[12, 24, 48, 96],
            output_size=(img_d, img_h, img_w),
            num_classes=num_classes,
            intermediate_channels=12
        )

    def forward(self, t1, t1ce, t2, flair):
        # Encoder Path - T1
        e1_t1 = self.enc1_t1(t1)
        d1_t1 = self.pool(e1_t1)
        e2_t1 = self.enc2_t1(d1_t1)
        d2_t1 = self.pool(e2_t1)
        e3_t1 = self.enc3_t1(d2_t1)
        d3_t1 = self.pool(e3_t1)
        e4_t1 = self.enc4_t1(d3_t1)

        # Encoder Path - T1CE
        e1_t1ce = self.enc1_t1ce(t1ce)
        d1_t1ce = self.pool(e1_t1ce)
        e2_t1ce = self.enc2_t1ce(d1_t1ce)
        d2_t1ce = self.pool(e2_t1ce)
        e3_t1ce = self.enc3_t1ce(d2_t1ce)
        d3_t1ce = self.pool(e3_t1ce)
        e4_t1ce = self.enc4_t1ce(d3_t1ce)

        # Encoder Path - T2
        e1_t2 = self.enc1_t2(t2)
        d1_t2 = self.pool(e1_t2)
        e2_t2 = self.enc2_t2(d1_t2)
        d2_t2 = self.pool(e2_t2)
        e3_t2 = self.enc3_t2(d2_t2)
        d3_t2 = self.pool(e3_t2)
        e4_t2 = self.enc4_t2(d3_t2)

        # Encoder Path - FLAIR
        e1_flair = self.enc1_flair(flair)
        d1_flair = self.pool(e1_flair)
        e2_flair = self.enc2_flair(d1_flair)
        d2_flair = self.pool(e2_flair)
        e3_flair = self.enc3_flair(d2_flair)
        d3_flair = self.pool(e3_flair)
        e4_flair = self.enc4_flair(d3_flair)

        # Merge encoder outputs
        merged_e4 = torch.cat([e4_t1, e4_t1ce, e4_t2, e4_flair], dim=1)      # 96*4 = 384
        merged_e3 = torch.cat([e3_t1, e3_t1ce, e3_t2, e3_flair], dim=1)      # 48*4 = 192
        merged_e2 = torch.cat([e2_t1, e2_t1ce, e2_t2, e2_flair], dim=1)      # 24*4 = 96
        merged_e1 = torch.cat([e1_t1, e1_t1ce, e1_t2, e1_flair], dim=1)      # 12*4 = 48

        # Bottleneck
        bottleneck = self.bottleneck(merged_e4)

        # Decoder Path
        merge5 = torch.cat([bottleneck, merged_e4], dim=1)
        c5 = self.dec1(merge5)

        up6 = self.upconv1(c5)
        merge6 = torch.cat([up6, merged_e3], dim=1)
        c6 = self.dec2(merge6)

        up7 = self.upconv2(c6)
        merge7 = torch.cat([up7, merged_e2], dim=1)
        c7 = self.dec3(merge7)

        up8 = self.upconv3(c7)
        merge8 = torch.cat([up8, merged_e1], dim=1)
        c8 = self.dec4(merge8)

        # Multi-Scale Feature Fusion
        output = self.fusion([c8, c7, c6, c5])
        return output


class base_AEMF(nn.Module):

    def __init__(self, img_h, img_w, img_d, in_channels=1, num_classes=4):
        super().__init__()

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoders
        self.enc1_t1 = DDSCR(in_channels, 12)
        self.enc2_t1 = DDSCR(12, 24)
        self.enc3_t1 = DDSCR(24, 48)
        self.enc4_t1 = DDSCR(48, 96)

        self.enc1_t1ce = DDSCR(in_channels, 12)
        self.enc2_t1ce = DDSCR(12, 24)
        self.enc3_t1ce = DDSCR(24, 48)
        self.enc4_t1ce = DDSCR(48, 96)

        self.enc1_t2 = DDSCR(in_channels, 12)
        self.enc2_t2 = DDSCR(12, 24)
        self.enc3_t2 = DDSCR(24, 48)
        self.enc4_t2 = DDSCR(48, 96)

        self.enc1_flair = DDSCR(in_channels, 12)
        self.enc2_flair = DDSCR(12, 24)
        self.enc3_flair = DDSCR(24, 48)
        self.enc4_flair = DDSCR(48, 96)

        # E-CMF REPLACES bottleneck (no LBB)
        self.aemf = AEMF(in_channels=96, reduction=8)  # Output: 96 channels

        # Decoder
        self.dec1 = DDSCR(480, 96)  # 96 + 384 = 480 (96 from ECMF + 96*4 skip)
        self.upconv1 = nn.ConvTranspose3d(96, 48, kernel_size=2, stride=2)

        self.dec2 = DDSCR(240, 48)  # 48 + 192 = 240 (48*4 concatenated)
        self.upconv2 = nn.ConvTranspose3d(48, 24, kernel_size=2, stride=2)

        self.dec3 = DDSCR(120, 24)  # 24 + 96 = 120 (24*4 concatenated)
        self.upconv3 = nn.ConvTranspose3d(24, 12, kernel_size=2, stride=2)

        self.dec4 = DDSCR(60, 12)  # 12 + 48 = 60 (12*4 concatenated)

        self.final = nn.Conv3d(12, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):
        # Encoder - T1
        e1_t1 = self.enc1_t1(t1)
        d1_t1 = self.pool(e1_t1)
        e2_t1 = self.enc2_t1(d1_t1)
        d2_t1 = self.pool(e2_t1)
        e3_t1 = self.enc3_t1(d2_t1)
        d3_t1 = self.pool(e3_t1)
        e4_t1 = self.enc4_t1(d3_t1)

        # Encoder - T1CE
        e1_t1ce = self.enc1_t1ce(t1ce)
        d1_t1ce = self.pool(e1_t1ce)
        e2_t1ce = self.enc2_t1ce(d1_t1ce)
        d2_t1ce = self.pool(e2_t1ce)
        e3_t1ce = self.enc3_t1ce(d2_t1ce)
        d3_t1ce = self.pool(e3_t1ce)
        e4_t1ce = self.enc4_t1ce(d3_t1ce)

        # Encoder - T2
        e1_t2 = self.enc1_t2(t2)
        d1_t2 = self.pool(e1_t2)
        e2_t2 = self.enc2_t2(d1_t2)
        d2_t2 = self.pool(e2_t2)
        e3_t2 = self.enc3_t2(d2_t2)
        d3_t2 = self.pool(e3_t2)
        e4_t2 = self.enc4_t2(d3_t2)

        # Encoder - FLAIR
        e1_flair = self.enc1_flair(flair)
        d1_flair = self.pool(e1_flair)
        e2_flair = self.enc2_flair(d1_flair)
        d2_flair = self.pool(e2_flair)
        e3_flair = self.enc3_flair(d2_flair)
        d3_flair = self.pool(e3_flair)
        e4_flair = self.enc4_flair(d3_flair)

        # Simple concatenation for ALL 4 skip connections
        merged_e1 = torch.cat([e1_t1, e1_t1ce, e1_t2, e1_flair], dim=1)  # 12*4 = 48
        merged_e2 = torch.cat([e2_t1, e2_t1ce, e2_t2, e2_flair], dim=1)  # 24*4 = 96
        merged_e3 = torch.cat([e3_t1, e3_t1ce, e3_t2, e3_flair], dim=1)  # 48*4 = 192
        merged_e4 = torch.cat([e4_t1, e4_t1ce, e4_t2, e4_flair], dim=1)  # 96*4 = 384

        # ECMF as bottleneck
        bottleneck = self.aemf(e4_t1, e4_t1ce, e4_t2, e4_flair)  # 96 channels

        # Decoder
        merge5 = torch.cat([bottleneck, merged_e4], dim=1)  # 96 + 384 = 480
        c5 = self.dec1(merge5)

        up6 = self.upconv1(c5)
        merge6 = torch.cat([up6, merged_e3], dim=1)  # 48 + 192 = 240
        c6 = self.dec2(merge6)

        up7 = self.upconv2(c6)
        merge7 = torch.cat([up7, merged_e2], dim=1)  # 24 + 96 = 120
        c7 = self.dec3(merge7)

        up8 = self.upconv3(c7)
        merge8 = torch.cat([up8, merged_e1], dim=1)  # 12 + 48 = 60
        c8 = self.dec4(merge8)

        out = self.final(c8)

        return out

class base_AEMF1(nn.Module):

    def __init__(self, img_h, img_w, img_d, in_channels=1, num_classes=4):
        super().__init__()

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoders
        self.enc1_t1 = DDSCR(in_channels, 12)
        self.enc2_t1 = DDSCR(12, 24)
        self.enc3_t1 = DDSCR(24, 48)
        self.enc4_t1 = DDSCR(48, 96)

        self.enc1_t1ce = DDSCR(in_channels, 12)
        self.enc2_t1ce = DDSCR(12, 24)
        self.enc3_t1ce = DDSCR(24, 48)
        self.enc4_t1ce = DDSCR(48, 96)

        self.enc1_t2 = DDSCR(in_channels, 12)
        self.enc2_t2 = DDSCR(12, 24)
        self.enc3_t2 = DDSCR(24, 48)
        self.enc4_t2 = DDSCR(48, 96)

        self.enc1_flair = DDSCR(in_channels, 12)
        self.enc2_flair = DDSCR(12, 24)
        self.enc3_flair = DDSCR(24, 48)
        self.enc4_flair = DDSCR(48, 96)

        # E-CMF REPLACES bottleneck (no LBB)
        self.aemf = AEMF1(in_channels=96, reduction=8)  # Output: 96 channels

        # Decoder
        self.dec1 = DDSCR(480, 96)  # 96 + 384 = 480 (96 from ECMF + 96*4 skip)
        self.upconv1 = nn.ConvTranspose3d(96, 48, kernel_size=2, stride=2)

        self.dec2 = DDSCR(240, 48)  # 48 + 192 = 240 (48*4 concatenated)
        self.upconv2 = nn.ConvTranspose3d(48, 24, kernel_size=2, stride=2)

        self.dec3 = DDSCR(120, 24)  # 24 + 96 = 120 (24*4 concatenated)
        self.upconv3 = nn.ConvTranspose3d(24, 12, kernel_size=2, stride=2)

        self.dec4 = DDSCR(60, 12)  # 12 + 48 = 60 (12*4 concatenated)

        self.final = nn.Conv3d(12, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):
        # Encoder - T1
        e1_t1 = self.enc1_t1(t1)
        d1_t1 = self.pool(e1_t1)
        e2_t1 = self.enc2_t1(d1_t1)
        d2_t1 = self.pool(e2_t1)
        e3_t1 = self.enc3_t1(d2_t1)
        d3_t1 = self.pool(e3_t1)
        e4_t1 = self.enc4_t1(d3_t1)

        # Encoder - T1CE
        e1_t1ce = self.enc1_t1ce(t1ce)
        d1_t1ce = self.pool(e1_t1ce)
        e2_t1ce = self.enc2_t1ce(d1_t1ce)
        d2_t1ce = self.pool(e2_t1ce)
        e3_t1ce = self.enc3_t1ce(d2_t1ce)
        d3_t1ce = self.pool(e3_t1ce)
        e4_t1ce = self.enc4_t1ce(d3_t1ce)

        # Encoder - T2
        e1_t2 = self.enc1_t2(t2)
        d1_t2 = self.pool(e1_t2)
        e2_t2 = self.enc2_t2(d1_t2)
        d2_t2 = self.pool(e2_t2)
        e3_t2 = self.enc3_t2(d2_t2)
        d3_t2 = self.pool(e3_t2)
        e4_t2 = self.enc4_t2(d3_t2)

        # Encoder - FLAIR
        e1_flair = self.enc1_flair(flair)
        d1_flair = self.pool(e1_flair)
        e2_flair = self.enc2_flair(d1_flair)
        d2_flair = self.pool(e2_flair)
        e3_flair = self.enc3_flair(d2_flair)
        d3_flair = self.pool(e3_flair)
        e4_flair = self.enc4_flair(d3_flair)

        # Simple concatenation for ALL 4 skip connections
        merged_e1 = torch.cat([e1_t1, e1_t1ce, e1_t2, e1_flair], dim=1)  # 12*4 = 48
        merged_e2 = torch.cat([e2_t1, e2_t1ce, e2_t2, e2_flair], dim=1)  # 24*4 = 96
        merged_e3 = torch.cat([e3_t1, e3_t1ce, e3_t2, e3_flair], dim=1)  # 48*4 = 192
        merged_e4 = torch.cat([e4_t1, e4_t1ce, e4_t2, e4_flair], dim=1)  # 96*4 = 384

        # ECMF as bottleneck
        bottleneck = self.aemf(e4_t1, e4_t1ce, e4_t2, e4_flair)  # 96 channels

        # Decoder
        merge5 = torch.cat([bottleneck, merged_e4], dim=1)  # 96 + 384 = 480
        c5 = self.dec1(merge5)

        up6 = self.upconv1(c5)
        merge6 = torch.cat([up6, merged_e3], dim=1)  # 48 + 192 = 240
        c6 = self.dec2(merge6)

        up7 = self.upconv2(c6)
        merge7 = torch.cat([up7, merged_e2], dim=1)  # 24 + 96 = 120
        c7 = self.dec3(merge7)

        up8 = self.upconv3(c7)
        merge8 = torch.cat([up8, merged_e1], dim=1)  # 12 + 48 = 60
        c8 = self.dec4(merge8)

        out = self.final(c8)
        return out

class base_AEMF2(nn.Module):

    def __init__(self, img_h, img_w, img_d, in_channels=1, num_classes=4):
        super().__init__()

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoders
        self.enc1_t1 = DDSCR(in_channels, 12)
        self.enc2_t1 = DDSCR(12, 24)
        self.enc3_t1 = DDSCR(24, 48)
        self.enc4_t1 = DDSCR(48, 96)

        self.enc1_t1ce = DDSCR(in_channels, 12)
        self.enc2_t1ce = DDSCR(12, 24)
        self.enc3_t1ce = DDSCR(24, 48)
        self.enc4_t1ce = DDSCR(48, 96)

        self.enc1_t2 = DDSCR(in_channels, 12)
        self.enc2_t2 = DDSCR(12, 24)
        self.enc3_t2 = DDSCR(24, 48)
        self.enc4_t2 = DDSCR(48, 96)

        self.enc1_flair = DDSCR(in_channels, 12)
        self.enc2_flair = DDSCR(12, 24)
        self.enc3_flair = DDSCR(24, 48)
        self.enc4_flair = DDSCR(48, 96)

        # E-CMF REPLACES bottleneck (no LBB)
        self.aemf = AEMF2(in_channels=96, reduction=8)  # Output: 96 channels

        # Decoder
        self.dec1 = DDSCR(480, 96)  # 96 + 384 = 480 (96 from ECMF + 96*4 skip)
        self.upconv1 = nn.ConvTranspose3d(96, 48, kernel_size=2, stride=2)

        self.dec2 = DDSCR(240, 48)  # 48 + 192 = 240 (48*4 concatenated)
        self.upconv2 = nn.ConvTranspose3d(48, 24, kernel_size=2, stride=2)

        self.dec3 = DDSCR(120, 24)  # 24 + 96 = 120 (24*4 concatenated)
        self.upconv3 = nn.ConvTranspose3d(24, 12, kernel_size=2, stride=2)

        self.dec4 = DDSCR(60, 12)  # 12 + 48 = 60 (12*4 concatenated)

        self.final = nn.Conv3d(12, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):
        # Encoder - T1
        e1_t1 = self.enc1_t1(t1)
        d1_t1 = self.pool(e1_t1)
        e2_t1 = self.enc2_t1(d1_t1)
        d2_t1 = self.pool(e2_t1)
        e3_t1 = self.enc3_t1(d2_t1)
        d3_t1 = self.pool(e3_t1)
        e4_t1 = self.enc4_t1(d3_t1)

        # Encoder - T1CE
        e1_t1ce = self.enc1_t1ce(t1ce)
        d1_t1ce = self.pool(e1_t1ce)
        e2_t1ce = self.enc2_t1ce(d1_t1ce)
        d2_t1ce = self.pool(e2_t1ce)
        e3_t1ce = self.enc3_t1ce(d2_t1ce)
        d3_t1ce = self.pool(e3_t1ce)
        e4_t1ce = self.enc4_t1ce(d3_t1ce)

        # Encoder - T2
        e1_t2 = self.enc1_t2(t2)
        d1_t2 = self.pool(e1_t2)
        e2_t2 = self.enc2_t2(d1_t2)
        d2_t2 = self.pool(e2_t2)
        e3_t2 = self.enc3_t2(d2_t2)
        d3_t2 = self.pool(e3_t2)
        e4_t2 = self.enc4_t2(d3_t2)

        # Encoder - FLAIR
        e1_flair = self.enc1_flair(flair)
        d1_flair = self.pool(e1_flair)
        e2_flair = self.enc2_flair(d1_flair)
        d2_flair = self.pool(e2_flair)
        e3_flair = self.enc3_flair(d2_flair)
        d3_flair = self.pool(e3_flair)
        e4_flair = self.enc4_flair(d3_flair)

        # Simple concatenation for ALL 4 skip connections
        merged_e1 = torch.cat([e1_t1, e1_t1ce, e1_t2, e1_flair], dim=1)  # 12*4 = 48
        merged_e2 = torch.cat([e2_t1, e2_t1ce, e2_t2, e2_flair], dim=1)  # 24*4 = 96
        merged_e3 = torch.cat([e3_t1, e3_t1ce, e3_t2, e3_flair], dim=1)  # 48*4 = 192
        merged_e4 = torch.cat([e4_t1, e4_t1ce, e4_t2, e4_flair], dim=1)  # 96*4 = 384

        # ECMF as bottleneck
        bottleneck = self.aemf(e4_t1, e4_t1ce, e4_t2, e4_flair)  # 96 channels

        # Decoder
        merge5 = torch.cat([bottleneck, merged_e4], dim=1)  # 96 + 384 = 480
        c5 = self.dec1(merge5)

        up6 = self.upconv1(c5)
        merge6 = torch.cat([up6, merged_e3], dim=1)  # 48 + 192 = 240
        c6 = self.dec2(merge6)

        up7 = self.upconv2(c6)
        merge7 = torch.cat([up7, merged_e2], dim=1)  # 24 + 96 = 120
        c7 = self.dec3(merge7)

        up8 = self.upconv3(c7)
        merge8 = torch.cat([up8, merged_e1], dim=1)  # 12 + 48 = 60
        c8 = self.dec4(merge8)

        out = self.final(c8)
        return out

class base_AEMF3(nn.Module):

    def __init__(self, img_h, img_w, img_d, in_channels=1, num_classes=4):
        super().__init__()

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoders
        self.enc1_t1 = DDSCR(in_channels, 12)
        self.enc2_t1 = DDSCR(12, 24)
        self.enc3_t1 = DDSCR(24, 48)
        self.enc4_t1 = DDSCR(48, 96)

        self.enc1_t1ce = DDSCR(in_channels, 12)
        self.enc2_t1ce = DDSCR(12, 24)
        self.enc3_t1ce = DDSCR(24, 48)
        self.enc4_t1ce = DDSCR(48, 96)

        self.enc1_t2 = DDSCR(in_channels, 12)
        self.enc2_t2 = DDSCR(12, 24)
        self.enc3_t2 = DDSCR(24, 48)
        self.enc4_t2 = DDSCR(48, 96)

        self.enc1_flair = DDSCR(in_channels, 12)
        self.enc2_flair = DDSCR(12, 24)
        self.enc3_flair = DDSCR(24, 48)
        self.enc4_flair = DDSCR(48, 96)

        # E-CMF REPLACES bottleneck (no LBB)
        self.aemf = AEMF_NoEntropy(in_channels=96, reduction=8)  # Output: 96 channels

        # Decoder
        self.dec1 = DDSCR(480, 96)  # 96 + 384 = 480 (96 from ECMF + 96*4 skip)
        self.upconv1 = nn.ConvTranspose3d(96, 48, kernel_size=2, stride=2)

        self.dec2 = DDSCR(240, 48)  # 48 + 192 = 240 (48*4 concatenated)
        self.upconv2 = nn.ConvTranspose3d(48, 24, kernel_size=2, stride=2)

        self.dec3 = DDSCR(120, 24)  # 24 + 96 = 120 (24*4 concatenated)
        self.upconv3 = nn.ConvTranspose3d(24, 12, kernel_size=2, stride=2)

        self.dec4 = DDSCR(60, 12)  # 12 + 48 = 60 (12*4 concatenated)

        self.final = nn.Conv3d(12, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):
        # Encoder - T1
        e1_t1 = self.enc1_t1(t1)
        d1_t1 = self.pool(e1_t1)
        e2_t1 = self.enc2_t1(d1_t1)
        d2_t1 = self.pool(e2_t1)
        e3_t1 = self.enc3_t1(d2_t1)
        d3_t1 = self.pool(e3_t1)
        e4_t1 = self.enc4_t1(d3_t1)

        # Encoder - T1CE
        e1_t1ce = self.enc1_t1ce(t1ce)
        d1_t1ce = self.pool(e1_t1ce)
        e2_t1ce = self.enc2_t1ce(d1_t1ce)
        d2_t1ce = self.pool(e2_t1ce)
        e3_t1ce = self.enc3_t1ce(d2_t1ce)
        d3_t1ce = self.pool(e3_t1ce)
        e4_t1ce = self.enc4_t1ce(d3_t1ce)

        # Encoder - T2
        e1_t2 = self.enc1_t2(t2)
        d1_t2 = self.pool(e1_t2)
        e2_t2 = self.enc2_t2(d1_t2)
        d2_t2 = self.pool(e2_t2)
        e3_t2 = self.enc3_t2(d2_t2)
        d3_t2 = self.pool(e3_t2)
        e4_t2 = self.enc4_t2(d3_t2)

        # Encoder - FLAIR
        e1_flair = self.enc1_flair(flair)
        d1_flair = self.pool(e1_flair)
        e2_flair = self.enc2_flair(d1_flair)
        d2_flair = self.pool(e2_flair)
        e3_flair = self.enc3_flair(d2_flair)
        d3_flair = self.pool(e3_flair)
        e4_flair = self.enc4_flair(d3_flair)

        # Simple concatenation for ALL 4 skip connections
        merged_e1 = torch.cat([e1_t1, e1_t1ce, e1_t2, e1_flair], dim=1)  # 12*4 = 48
        merged_e2 = torch.cat([e2_t1, e2_t1ce, e2_t2, e2_flair], dim=1)  # 24*4 = 96
        merged_e3 = torch.cat([e3_t1, e3_t1ce, e3_t2, e3_flair], dim=1)  # 48*4 = 192
        merged_e4 = torch.cat([e4_t1, e4_t1ce, e4_t2, e4_flair], dim=1)  # 96*4 = 384

        # ECMF as bottleneck
        bottleneck = self.aemf(e4_t1, e4_t1ce, e4_t2, e4_flair)  # 96 channels

        # Decoder
        merge5 = torch.cat([bottleneck, merged_e4], dim=1)  # 96 + 384 = 480
        c5 = self.dec1(merge5)

        up6 = self.upconv1(c5)
        merge6 = torch.cat([up6, merged_e3], dim=1)  # 48 + 192 = 240
        c6 = self.dec2(merge6)

        up7 = self.upconv2(c6)
        merge7 = torch.cat([up7, merged_e2], dim=1)  # 24 + 96 = 120
        c7 = self.dec3(merge7)

        up8 = self.upconv3(c7)
        merge8 = torch.cat([up8, merged_e1], dim=1)  # 12 + 48 = 60
        c8 = self.dec4(merge8)
        out = self.final(c8)
        return out


class DENet(nn.Module):

    def __init__(self, img_h, img_w, img_d, in_channels=1, num_classes=4):
        super().__init__()

        # Store original input size
        self.img_d = img_d
        self.img_h = img_h
        self.img_w = img_w

        # Max pooling layer
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder blocks for T1
        self.enc1_t1 = DDSCR(in_channels, 12)
        self.enc2_t1 = DDSCR(12, 24)
        self.enc3_t1 = DDSCR(24, 48)
        self.enc4_t1 = DDSCR(48, 96)

        # Encoder blocks for T1CE
        self.enc1_t1ce = DDSCR(in_channels, 12)
        self.enc2_t1ce = DDSCR(12, 24)
        self.enc3_t1ce = DDSCR(24, 48)
        self.enc4_t1ce = DDSCR(48, 96)

        # Encoder blocks for T2
        self.enc1_t2 = DDSCR(in_channels, 12)
        self.enc2_t2 = DDSCR(12, 24)
        self.enc3_t2 = DDSCR(24, 48)
        self.enc4_t2 = DDSCR(48, 96)

        # Encoder blocks for FLAIR
        self.enc1_flair = DDSCR(in_channels, 12)
        self.enc2_flair = DDSCR(12, 24)
        self.enc3_flair = DDSCR(24, 48)
        self.enc4_flair = DDSCR(48, 96)

        # Each DME encodes discriminative differences at each scale
        self.dme_skip1 = DME1(12, out_channels=24)  # 12 -> 24
        self.dme_skip2 = DME1(24, out_channels=48)  # 24 -> 48
        self.dme_skip3 = DME1(48, out_channels=96)  # 48 -> 96
        self.dme_skip4 = DME1(96, out_channels=192)  # 96 -> 192

        self.aemf = AEMF(in_channels=96, reduction=8)  # Output: 96 channels

        self.dec1 = DDSCR(288, 96)  # 96 (ECMF) + 192 (dme_skip4) = 288
        self.upconv1 = nn.ConvTranspose3d(96, 48, kernel_size=2, stride=2)

        self.dec2 = DDSCR(144, 48)  # 48 (upconv1) + 96 (dme_skip3) = 144
        self.upconv2 = nn.ConvTranspose3d(48, 24, kernel_size=2, stride=2)

        self.dec3 = DDSCR(72, 24)  # 24 (upconv2) + 48 (dme_skip2) = 72
        self.upconv3 = nn.ConvTranspose3d(24, 12, kernel_size=2, stride=2)

        self.dec4 = DDSCR(36, 12)  # 12 (upconv3) + 24 (dme_skip1) = 36

        self.fusion = PGAM(
            feature_channels=[12, 24, 48, 96],  # From finest to coarsest
            output_size=(img_d, img_h, img_w),
            num_classes=num_classes,
            intermediate_channels=12  # NEW parameter (matches your first layer)
        )

    def forward(self, t1, t1ce, t2, flair):

        # T1 pathway
        e1_t1 = self.enc1_t1(t1)
        d1_t1 = self.pool(e1_t1)

        e2_t1 = self.enc2_t1(d1_t1)
        d2_t1 = self.pool(e2_t1)

        e3_t1 = self.enc3_t1(d2_t1)
        d3_t1 = self.pool(e3_t1)

        e4_t1 = self.enc4_t1(d3_t1)

        # T1CE pathway
        e1_t1ce = self.enc1_t1ce(t1ce)
        d1_t1ce = self.pool(e1_t1ce)
        e2_t1ce = self.enc2_t1ce(d1_t1ce)
        d2_t1ce = self.pool(e2_t1ce)
        e3_t1ce = self.enc3_t1ce(d2_t1ce)
        d3_t1ce = self.pool(e3_t1ce)
        e4_t1ce = self.enc4_t1ce(d3_t1ce)

        # T2 pathway
        e1_t2 = self.enc1_t2(t2)
        d1_t2 = self.pool(e1_t2)
        e2_t2 = self.enc2_t2(d1_t2)
        d2_t2 = self.pool(e2_t2)
        e3_t2 = self.enc3_t2(d2_t2)
        d3_t2 = self.pool(e3_t2)
        e4_t2 = self.enc4_t2(d3_t2)

        # FLAIR pathway
        e1_flair = self.enc1_flair(flair)
        d1_flair = self.pool(e1_flair)
        e2_flair = self.enc2_flair(d1_flair)
        d2_flair = self.pool(e2_flair)
        e3_flair = self.enc3_flair(d2_flair)
        d3_flair = self.pool(e3_flair)
        e4_flair = self.enc4_flair(d3_flair)

        # DME encodes discriminative inter-modal boundaries
        dme_e1 = self.dme_skip1(e1_t1, e1_t1ce, e1_t2, e1_flair)  # [B, 24, D, H, W]
        dme_e2 = self.dme_skip2(e2_t1, e2_t1ce, e2_t2, e2_flair)  # [B, 48, D/2, H/2, W/2]
        dme_e3 = self.dme_skip3(e3_t1, e3_t1ce, e3_t2, e3_flair)  # [B, 96, D/4, H/4, W/4]
        dme_e4 = self.dme_skip4(e4_t1, e4_t1ce, e4_t2, e4_flair)  # [B, 192, D/8, H/8, W/8]

        bottleneck = self.aemf(e4_t1, e4_t1ce, e4_t2, e4_flair)  # [B, 96, D/8, H/8, W/8]

        # Level 1: Bottleneck + DME skip 4
        merge5 = torch.cat([bottleneck, dme_e4], dim=1)  # 96 + 192 = 288
        c5 = self.dec1(merge5)  # [B, 96, D/8, H/8, W/8]

        # Level 2: Upsample + DME skip 3
        up6 = self.upconv1(c5)  # [B, 48, D/4, H/4, W/4]
        merge6 = torch.cat([up6, dme_e3], dim=1)  # 48 + 96 = 144
        c6 = self.dec2(merge6)  # [B, 48, D/4, H/4, W/4]

        # Level 3: Upsample + DME skip 2
        up7 = self.upconv2(c6)  # [B, 24, D/2, H/2, W/2]
        merge7 = torch.cat([up7, dme_e2], dim=1)  # 24 + 48 = 72
        c7 = self.dec3(merge7)  # [B, 24, D/2, H/2, W/2]

        # Level 4: Upsample + DME skip 1
        up8 = self.upconv3(c7)  # [B, 12, D, H, W]
        merge8 = torch.cat([up8, dme_e1], dim=1)  # 12 + 24 = 36
        c8 = self.dec4(merge8)  # [B, 12, D, H, W]

        # Pass features from finest to coarsest: [c8, c7, c6, c5, bottleneck]
        output = self.fusion([c8, c7, c6, c5])
        return output

    def get_dme_visualizations(self, t1, t1ce, t2, flair, return_all_levels=False):
        """
        Extract DME difference maps at all encoder levels for visualization

        Args:
            t1, t1ce, t2, flair: [B, 1, D, H, W] input modalities
            return_all_levels: If True, return maps from all 4 levels

        Returns:
            Dictionary containing difference maps at each level
        """
        with torch.no_grad():
            # Encode all modalities to all levels
            # Level 1
            e1_t1 = self.enc1_t1(t1)
            e1_t1ce = self.enc1_t1ce(t1ce)
            e1_t2 = self.enc1_t2(t2)
            e1_flair = self.enc1_flair(flair)

            if return_all_levels:
                # Level 2
                d1_t1 = self.pool(e1_t1)
                e2_t1 = self.enc2_t1(d1_t1)
                d1_t1ce = self.pool(e1_t1ce)
                e2_t1ce = self.enc2_t1ce(d1_t1ce)
                d1_t2 = self.pool(e1_t2)
                e2_t2 = self.enc2_t2(d1_t2)
                d1_flair = self.pool(e1_flair)
                e2_flair = self.enc2_flair(d1_flair)

                # Level 3
                d2_t1 = self.pool(e2_t1)
                e3_t1 = self.enc3_t1(d2_t1)
                d2_t1ce = self.pool(e2_t1ce)
                e3_t1ce = self.enc3_t1ce(d2_t1ce)
                d2_t2 = self.pool(e2_t2)
                e3_t2 = self.enc3_t2(d2_t2)
                d2_flair = self.pool(e2_flair)
                e3_flair = self.enc3_flair(d2_flair)

                # Level 4
                d3_t1 = self.pool(e3_t1)
                e4_t1 = self.enc4_t1(d3_t1)
                d3_t1ce = self.pool(e3_t1ce)
                e4_t1ce = self.enc4_t1ce(d3_t1ce)
                d3_t2 = self.pool(e3_t2)
                e4_t2 = self.enc4_t2(d3_t2)
                d3_flair = self.pool(e3_flair)
                e4_flair = self.enc4_flair(d3_flair)

            # Get difference maps
            result = {
                'level1': {
                    'maps': self.dme_skip1.get_difference_maps(e1_t1, e1_t1ce, e1_t2, e1_flair),
                    'spatial_size': f"{self.img_d}x{self.img_h}x{self.img_w}",
                    'channels': 12
                },
                'config': {
                    'mode': self.dme_skip1.diff_mode,
                    'num_diffs': self.dme_skip1.num_diffs,
                    'learnable_weights': self.dme_skip1.use_learnable_weights
                }
            }

            if return_all_levels:
                result['level2'] = {
                    'maps': self.dme_skip2.get_difference_maps(e2_t1, e2_t1ce, e2_t2, e2_flair),
                    'spatial_size': f"{self.img_d // 2}x{self.img_h // 2}x{self.img_w // 2}",
                    'channels': 24
                }
                result['level3'] = {
                    'maps': self.dme_skip3.get_difference_maps(e3_t1, e3_t1ce, e3_t2, e3_flair),
                    'spatial_size': f"{self.img_d // 4}x{self.img_h // 4}x{self.img_w // 4}",
                    'channels': 48
                }
                result['level4'] = {
                    'maps': self.dme_skip4.get_difference_maps(e4_t1, e4_t1ce, e4_t2, e4_flair),
                    'spatial_size': f"{self.img_d // 8}x{self.img_h // 8}x{self.img_w // 8}",
                    'channels': 96
                }

            return result

    def get_learned_diff_weights(self):
        """
        Extract learned importance weights from all DME modules

        Returns:
            Dictionary with weights at each scale
        """
        return {
            'skip1': self.dme_skip1.get_learned_weights(),
            'skip2': self.dme_skip2.get_learned_weights(),
            'skip3': self.dme_skip3.get_learned_weights(),
            'skip4': self.dme_skip4.get_learned_weights(),
            'config': {
                'mode': self.dme_skip1.diff_mode,
                'learnable': self.dme_skip1.use_learnable_weights
            }
        }

class base_all2(nn.Module):

    def __init__(self, img_h, img_w, img_d, in_channels=1, num_classes=4):
        super().__init__()

        # Store original input size
        self.img_d = img_d
        self.img_h = img_h
        self.img_w = img_w

        # Max pooling layer
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder blocks for T1
        self.enc1_t1 = DDSCR(in_channels, 12)
        self.enc2_t1 = DDSCR(12, 24)
        self.enc3_t1 = DDSCR(24, 48)
        self.enc4_t1 = DDSCR(48, 96)

        # Encoder blocks for T1CE
        self.enc1_t1ce = DDSCR(in_channels, 12)
        self.enc2_t1ce = DDSCR(12, 24)
        self.enc3_t1ce = DDSCR(24, 48)
        self.enc4_t1ce = DDSCR(48, 96)

        # Encoder blocks for T2
        self.enc1_t2 = DDSCR(in_channels, 12)
        self.enc2_t2 = DDSCR(12, 24)
        self.enc3_t2 = DDSCR(24, 48)
        self.enc4_t2 = DDSCR(48, 96)

        # Encoder blocks for FLAIR
        self.enc1_flair = DDSCR(in_channels, 12)
        self.enc2_flair = DDSCR(12, 24)
        self.enc3_flair = DDSCR(24, 48)
        self.enc4_flair = DDSCR(48, 96)

        # Each DME encodes discriminative differences at each scale
        self.dme_skip1 = DME(12, out_channels=24)  # 12 -> 24
        self.dme_skip2 = DME(24, out_channels=48)  # 24 -> 48
        self.dme_skip3 = DME(48, out_channels=96)  # 48 -> 96
        self.dme_skip4 = DME(96, out_channels=192)  # 96 -> 192

        self.aemf = AEMF(in_channels=96, reduction=8)  # Output: 96 channels

        self.dec1 = DDSCR(288, 96)  # 96 (ECMF) + 192 (dme_skip4) = 288
        self.upconv1 = nn.ConvTranspose3d(96, 48, kernel_size=2, stride=2)

        self.dec2 = DDSCR(144, 48)  # 48 (upconv1) + 96 (dme_skip3) = 144
        self.upconv2 = nn.ConvTranspose3d(48, 24, kernel_size=2, stride=2)

        self.dec3 = DDSCR(72, 24)  # 24 (upconv2) + 48 (dme_skip2) = 72
        self.upconv3 = nn.ConvTranspose3d(24, 12, kernel_size=2, stride=2)

        self.dec4 = DDSCR(36, 12)  # 12 (upconv3) + 24 (dme_skip1) = 36

        self.fusion = PGAM(
            feature_channels=[12, 24, 48, 96],  # From finest to coarsest
            output_size=(img_d, img_h, img_w),
            num_classes=num_classes,
            intermediate_channels=12  # NEW parameter (matches your first layer)
        )

    def forward(self, t1, t1ce, t2, flair):

        # T1 pathway
        e1_t1 = self.enc1_t1(t1)
        d1_t1 = self.pool(e1_t1)

        e2_t1 = self.enc2_t1(d1_t1)
        d2_t1 = self.pool(e2_t1)

        e3_t1 = self.enc3_t1(d2_t1)
        d3_t1 = self.pool(e3_t1)

        e4_t1 = self.enc4_t1(d3_t1)

        # T1CE pathway
        e1_t1ce = self.enc1_t1ce(t1ce)
        d1_t1ce = self.pool(e1_t1ce)
        e2_t1ce = self.enc2_t1ce(d1_t1ce)
        d2_t1ce = self.pool(e2_t1ce)
        e3_t1ce = self.enc3_t1ce(d2_t1ce)
        d3_t1ce = self.pool(e3_t1ce)
        e4_t1ce = self.enc4_t1ce(d3_t1ce)

        # T2 pathway
        e1_t2 = self.enc1_t2(t2)
        d1_t2 = self.pool(e1_t2)
        e2_t2 = self.enc2_t2(d1_t2)
        d2_t2 = self.pool(e2_t2)
        e3_t2 = self.enc3_t2(d2_t2)
        d3_t2 = self.pool(e3_t2)
        e4_t2 = self.enc4_t2(d3_t2)

        # FLAIR pathway
        e1_flair = self.enc1_flair(flair)
        d1_flair = self.pool(e1_flair)
        e2_flair = self.enc2_flair(d1_flair)
        d2_flair = self.pool(e2_flair)
        e3_flair = self.enc3_flair(d2_flair)
        d3_flair = self.pool(e3_flair)
        e4_flair = self.enc4_flair(d3_flair)

        # DME encodes discriminative inter-modal boundaries
        dme_e1 = self.dme_skip1(e1_t1, e1_t1ce, e1_t2, e1_flair)  # [B, 24, D, H, W]
        dme_e2 = self.dme_skip2(e2_t1, e2_t1ce, e2_t2, e2_flair)  # [B, 48, D/2, H/2, W/2]
        dme_e3 = self.dme_skip3(e3_t1, e3_t1ce, e3_t2, e3_flair)  # [B, 96, D/4, H/4, W/4]
        dme_e4 = self.dme_skip4(e4_t1, e4_t1ce, e4_t2, e4_flair)  # [B, 192, D/8, H/8, W/8]

        bottleneck = self.aemf(e4_t1, e4_t1ce, e4_t2, e4_flair)  # [B, 96, D/8, H/8, W/8]

        # Level 1: Bottleneck + DME skip 4
        merge5 = torch.cat([bottleneck, dme_e4], dim=1)  # 96 + 192 = 288
        c5 = self.dec1(merge5)  # [B, 96, D/8, H/8, W/8]

        # Level 2: Upsample + DME skip 3
        up6 = self.upconv1(c5)  # [B, 48, D/4, H/4, W/4]
        merge6 = torch.cat([up6, dme_e3], dim=1)  # 48 + 96 = 144
        c6 = self.dec2(merge6)  # [B, 48, D/4, H/4, W/4]

        # Level 3: Upsample + DME skip 2
        up7 = self.upconv2(c6)  # [B, 24, D/2, H/2, W/2]
        merge7 = torch.cat([up7, dme_e2], dim=1)  # 24 + 48 = 72
        c7 = self.dec3(merge7)  # [B, 24, D/2, H/2, W/2]

        # Level 4: Upsample + DME skip 1
        up8 = self.upconv3(c7)  # [B, 12, D, H, W]
        merge8 = torch.cat([up8, dme_e1], dim=1)  # 12 + 24 = 36
        c8 = self.dec4(merge8)  # [B, 12, D, H, W]

        # Pass features from finest to coarsest: [c8, c7, c6, c5, bottleneck]
        output = self.fusion([c8, c7, c6, c5])
        return output

    def get_dme_visualizations(self, t1, t1ce, t2, flair, return_all_levels=False):
        """
        Extract DME difference maps at all encoder levels for visualization

        Args:
            t1, t1ce, t2, flair: [B, 1, D, H, W] input modalities
            return_all_levels: If True, return maps from all 4 levels

        Returns:
            Dictionary containing difference maps at each level
        """
        with torch.no_grad():
            # Encode all modalities to all levels
            # Level 1
            e1_t1 = self.enc1_t1(t1)
            e1_t1ce = self.enc1_t1ce(t1ce)
            e1_t2 = self.enc1_t2(t2)
            e1_flair = self.enc1_flair(flair)

            if return_all_levels:
                # Level 2
                d1_t1 = self.pool(e1_t1)
                e2_t1 = self.enc2_t1(d1_t1)
                d1_t1ce = self.pool(e1_t1ce)
                e2_t1ce = self.enc2_t1ce(d1_t1ce)
                d1_t2 = self.pool(e1_t2)
                e2_t2 = self.enc2_t2(d1_t2)
                d1_flair = self.pool(e1_flair)
                e2_flair = self.enc2_flair(d1_flair)

                # Level 3
                d2_t1 = self.pool(e2_t1)
                e3_t1 = self.enc3_t1(d2_t1)
                d2_t1ce = self.pool(e2_t1ce)
                e3_t1ce = self.enc3_t1ce(d2_t1ce)
                d2_t2 = self.pool(e2_t2)
                e3_t2 = self.enc3_t2(d2_t2)
                d2_flair = self.pool(e2_flair)
                e3_flair = self.enc3_flair(d2_flair)

                # Level 4
                d3_t1 = self.pool(e3_t1)
                e4_t1 = self.enc4_t1(d3_t1)
                d3_t1ce = self.pool(e3_t1ce)
                e4_t1ce = self.enc4_t1ce(d3_t1ce)
                d3_t2 = self.pool(e3_t2)
                e4_t2 = self.enc4_t2(d3_t2)
                d3_flair = self.pool(e3_flair)
                e4_flair = self.enc4_flair(d3_flair)

            # Get difference maps
            result = {
                'level1': {
                    'maps': self.dme_skip1.get_difference_maps(e1_t1, e1_t1ce, e1_t2, e1_flair),
                    'spatial_size': f"{self.img_d}x{self.img_h}x{self.img_w}",
                    'channels': 12
                },
                'config': {
                    'mode': self.dme_skip1.diff_mode,
                    'num_diffs': self.dme_skip1.num_diffs,
                    'learnable_weights': self.dme_skip1.use_learnable_weights
                }
            }

            if return_all_levels:
                result['level2'] = {
                    'maps': self.dme_skip2.get_difference_maps(e2_t1, e2_t1ce, e2_t2, e2_flair),
                    'spatial_size': f"{self.img_d // 2}x{self.img_h // 2}x{self.img_w // 2}",
                    'channels': 24
                }
                result['level3'] = {
                    'maps': self.dme_skip3.get_difference_maps(e3_t1, e3_t1ce, e3_t2, e3_flair),
                    'spatial_size': f"{self.img_d // 4}x{self.img_h // 4}x{self.img_w // 4}",
                    'channels': 48
                }
                result['level4'] = {
                    'maps': self.dme_skip4.get_difference_maps(e4_t1, e4_t1ce, e4_t2, e4_flair),
                    'spatial_size': f"{self.img_d // 8}x{self.img_h // 8}x{self.img_w // 8}",
                    'channels': 96
                }

            return result

    def get_learned_diff_weights(self):
        """
        Extract learned importance weights from all DME modules

        Returns:
            Dictionary with weights at each scale
        """
        return {
            'skip1': self.dme_skip1.get_learned_weights(),
            'skip2': self.dme_skip2.get_learned_weights(),
            'skip3': self.dme_skip3.get_learned_weights(),
            'skip4': self.dme_skip4.get_learned_weights(),
            'config': {
                'mode': self.dme_skip1.diff_mode,
                'learnable': self.dme_skip1.use_learnable_weights
            }
        }


class base_all3(nn.Module):

    def __init__(self, img_h, img_w, img_d, in_channels=1, num_classes=4):
        super().__init__()

        # Store original input size
        self.img_d = img_d
        self.img_h = img_h
        self.img_w = img_w

        # Max pooling layer
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder blocks for T1
        self.enc1_t1 = DDSCR(in_channels, 12)
        self.enc2_t1 = DDSCR(12, 24)
        self.enc3_t1 = DDSCR(24, 48)
        self.enc4_t1 = DDSCR(48, 96)

        # Encoder blocks for T1CE
        self.enc1_t1ce = DDSCR(in_channels, 12)
        self.enc2_t1ce = DDSCR(12, 24)
        self.enc3_t1ce = DDSCR(24, 48)
        self.enc4_t1ce = DDSCR(48, 96)

        # Encoder blocks for T2
        self.enc1_t2 = DDSCR(in_channels, 12)
        self.enc2_t2 = DDSCR(12, 24)
        self.enc3_t2 = DDSCR(24, 48)
        self.enc4_t2 = DDSCR(48, 96)

        # Encoder blocks for FLAIR
        self.enc1_flair = DDSCR(in_channels, 12)
        self.enc2_flair = DDSCR(12, 24)
        self.enc3_flair = DDSCR(24, 48)
        self.enc4_flair = DDSCR(48, 96)

        # Each DME encodes discriminative differences at each scale
        self.dme_skip1 = DME(12, out_channels=24)  # 12 -> 24
        self.dme_skip2 = DME(24, out_channels=48)  # 24 -> 48
        self.dme_skip3 = DME(48, out_channels=96)  # 48 -> 96
        self.dme_skip4 = DME(96, out_channels=192)  # 96 -> 192

        self.aemf = AEMF(in_channels=96, reduction=8)  # Output: 96 channels

        self.dec1 = DDSCR(288, 96)  # 96 (ECMF) + 192 (dme_skip4) = 288
        self.upconv1 = nn.ConvTranspose3d(96, 48, kernel_size=2, stride=2)

        self.dec2 = DDSCR(144, 48)  # 48 (upconv1) + 96 (dme_skip3) = 144
        self.upconv2 = nn.ConvTranspose3d(48, 24, kernel_size=2, stride=2)

        self.dec3 = DDSCR(72, 24)  # 24 (upconv2) + 48 (dme_skip2) = 72
        self.upconv3 = nn.ConvTranspose3d(24, 12, kernel_size=2, stride=2)

        self.dec4 = DDSCR(36, 12)  # 12 (upconv3) + 24 (dme_skip1) = 36

        self.fusion = PGAM2(
            feature_channels=[12, 24, 48, 96],  # From finest to coarsest
            output_size=(img_d, img_h, img_w),
            num_classes=num_classes,
            intermediate_channels=12  # NEW parameter (matches your first layer)
        )

    def forward(self, t1, t1ce, t2, flair):

        # T1 pathway
        e1_t1 = self.enc1_t1(t1)
        d1_t1 = self.pool(e1_t1)

        e2_t1 = self.enc2_t1(d1_t1)
        d2_t1 = self.pool(e2_t1)

        e3_t1 = self.enc3_t1(d2_t1)
        d3_t1 = self.pool(e3_t1)

        e4_t1 = self.enc4_t1(d3_t1)

        # T1CE pathway
        e1_t1ce = self.enc1_t1ce(t1ce)
        d1_t1ce = self.pool(e1_t1ce)
        e2_t1ce = self.enc2_t1ce(d1_t1ce)
        d2_t1ce = self.pool(e2_t1ce)
        e3_t1ce = self.enc3_t1ce(d2_t1ce)
        d3_t1ce = self.pool(e3_t1ce)
        e4_t1ce = self.enc4_t1ce(d3_t1ce)

        # T2 pathway
        e1_t2 = self.enc1_t2(t2)
        d1_t2 = self.pool(e1_t2)
        e2_t2 = self.enc2_t2(d1_t2)
        d2_t2 = self.pool(e2_t2)
        e3_t2 = self.enc3_t2(d2_t2)
        d3_t2 = self.pool(e3_t2)
        e4_t2 = self.enc4_t2(d3_t2)

        # FLAIR pathway
        e1_flair = self.enc1_flair(flair)
        d1_flair = self.pool(e1_flair)
        e2_flair = self.enc2_flair(d1_flair)
        d2_flair = self.pool(e2_flair)
        e3_flair = self.enc3_flair(d2_flair)
        d3_flair = self.pool(e3_flair)
        e4_flair = self.enc4_flair(d3_flair)

        # DME encodes discriminative inter-modal boundaries
        dme_e1 = self.dme_skip1(e1_t1, e1_t1ce, e1_t2, e1_flair)  # [B, 24, D, H, W]
        dme_e2 = self.dme_skip2(e2_t1, e2_t1ce, e2_t2, e2_flair)  # [B, 48, D/2, H/2, W/2]
        dme_e3 = self.dme_skip3(e3_t1, e3_t1ce, e3_t2, e3_flair)  # [B, 96, D/4, H/4, W/4]
        dme_e4 = self.dme_skip4(e4_t1, e4_t1ce, e4_t2, e4_flair)  # [B, 192, D/8, H/8, W/8]

        bottleneck = self.aemf(e4_t1, e4_t1ce, e4_t2, e4_flair)  # [B, 96, D/8, H/8, W/8]

        # Level 1: Bottleneck + DME skip 4
        merge5 = torch.cat([bottleneck, dme_e4], dim=1)  # 96 + 192 = 288
        c5 = self.dec1(merge5)  # [B, 96, D/8, H/8, W/8]

        # Level 2: Upsample + DME skip 3
        up6 = self.upconv1(c5)  # [B, 48, D/4, H/4, W/4]
        merge6 = torch.cat([up6, dme_e3], dim=1)  # 48 + 96 = 144
        c6 = self.dec2(merge6)  # [B, 48, D/4, H/4, W/4]

        # Level 3: Upsample + DME skip 2
        up7 = self.upconv2(c6)  # [B, 24, D/2, H/2, W/2]
        merge7 = torch.cat([up7, dme_e2], dim=1)  # 24 + 48 = 72
        c7 = self.dec3(merge7)  # [B, 24, D/2, H/2, W/2]

        # Level 4: Upsample + DME skip 1
        up8 = self.upconv3(c7)  # [B, 12, D, H, W]
        merge8 = torch.cat([up8, dme_e1], dim=1)  # 12 + 24 = 36
        c8 = self.dec4(merge8)  # [B, 12, D, H, W]

        # Pass features from finest to coarsest: [c8, c7, c6, c5, bottleneck]
        output = self.fusion([c8, c7, c6, c5])
        return output

    def get_dme_visualizations(self, t1, t1ce, t2, flair, return_all_levels=False):
        """
        Extract DME difference maps at all encoder levels for visualization

        Args:
            t1, t1ce, t2, flair: [B, 1, D, H, W] input modalities
            return_all_levels: If True, return maps from all 4 levels

        Returns:
            Dictionary containing difference maps at each level
        """
        with torch.no_grad():
            # Encode all modalities to all levels
            # Level 1
            e1_t1 = self.enc1_t1(t1)
            e1_t1ce = self.enc1_t1ce(t1ce)
            e1_t2 = self.enc1_t2(t2)
            e1_flair = self.enc1_flair(flair)

            if return_all_levels:
                # Level 2
                d1_t1 = self.pool(e1_t1)
                e2_t1 = self.enc2_t1(d1_t1)
                d1_t1ce = self.pool(e1_t1ce)
                e2_t1ce = self.enc2_t1ce(d1_t1ce)
                d1_t2 = self.pool(e1_t2)
                e2_t2 = self.enc2_t2(d1_t2)
                d1_flair = self.pool(e1_flair)
                e2_flair = self.enc2_flair(d1_flair)

                # Level 3
                d2_t1 = self.pool(e2_t1)
                e3_t1 = self.enc3_t1(d2_t1)
                d2_t1ce = self.pool(e2_t1ce)
                e3_t1ce = self.enc3_t1ce(d2_t1ce)
                d2_t2 = self.pool(e2_t2)
                e3_t2 = self.enc3_t2(d2_t2)
                d2_flair = self.pool(e2_flair)
                e3_flair = self.enc3_flair(d2_flair)

                # Level 4
                d3_t1 = self.pool(e3_t1)
                e4_t1 = self.enc4_t1(d3_t1)
                d3_t1ce = self.pool(e3_t1ce)
                e4_t1ce = self.enc4_t1ce(d3_t1ce)
                d3_t2 = self.pool(e3_t2)
                e4_t2 = self.enc4_t2(d3_t2)
                d3_flair = self.pool(e3_flair)
                e4_flair = self.enc4_flair(d3_flair)

            # Get difference maps
            result = {
                'level1': {
                    'maps': self.dme_skip1.get_difference_maps(e1_t1, e1_t1ce, e1_t2, e1_flair),
                    'spatial_size': f"{self.img_d}x{self.img_h}x{self.img_w}",
                    'channels': 12
                },
                'config': {
                    'mode': self.dme_skip1.diff_mode,
                    'num_diffs': self.dme_skip1.num_diffs,
                    'learnable_weights': self.dme_skip1.use_learnable_weights
                }
            }

            if return_all_levels:
                result['level2'] = {
                    'maps': self.dme_skip2.get_difference_maps(e2_t1, e2_t1ce, e2_t2, e2_flair),
                    'spatial_size': f"{self.img_d // 2}x{self.img_h // 2}x{self.img_w // 2}",
                    'channels': 24
                }
                result['level3'] = {
                    'maps': self.dme_skip3.get_difference_maps(e3_t1, e3_t1ce, e3_t2, e3_flair),
                    'spatial_size': f"{self.img_d // 4}x{self.img_h // 4}x{self.img_w // 4}",
                    'channels': 48
                }
                result['level4'] = {
                    'maps': self.dme_skip4.get_difference_maps(e4_t1, e4_t1ce, e4_t2, e4_flair),
                    'spatial_size': f"{self.img_d // 8}x{self.img_h // 8}x{self.img_w // 8}",
                    'channels': 96
                }

            return result

    def get_learned_diff_weights(self):
        """
        Extract learned importance weights from all DME modules

        Returns:
            Dictionary with weights at each scale
        """
        return {
            'skip1': self.dme_skip1.get_learned_weights(),
            'skip2': self.dme_skip2.get_learned_weights(),
            'skip3': self.dme_skip3.get_learned_weights(),
            'skip4': self.dme_skip4.get_learned_weights(),
            'config': {
                'mode': self.dme_skip1.diff_mode,
                'learnable': self.dme_skip1.use_learnable_weights
            }
        }

class base_all4(nn.Module):

    def __init__(self, img_h, img_w, img_d, in_channels=1, num_classes=4):
        super().__init__()

        # Store original input size
        self.img_d = img_d
        self.img_h = img_h
        self.img_w = img_w

        # Max pooling layer
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder blocks for T1
        self.enc1_t1 = DDSCR(in_channels, 12)
        self.enc2_t1 = DDSCR(12, 24)
        self.enc3_t1 = DDSCR(24, 48)
        self.enc4_t1 = DDSCR(48, 96)

        # Encoder blocks for T1CE
        self.enc1_t1ce = DDSCR(in_channels, 12)
        self.enc2_t1ce = DDSCR(12, 24)
        self.enc3_t1ce = DDSCR(24, 48)
        self.enc4_t1ce = DDSCR(48, 96)

        # Encoder blocks for T2
        self.enc1_t2 = DDSCR(in_channels, 12)
        self.enc2_t2 = DDSCR(12, 24)
        self.enc3_t2 = DDSCR(24, 48)
        self.enc4_t2 = DDSCR(48, 96)

        # Encoder blocks for FLAIR
        self.enc1_flair = DDSCR(in_channels, 12)
        self.enc2_flair = DDSCR(12, 24)
        self.enc3_flair = DDSCR(24, 48)
        self.enc4_flair = DDSCR(48, 96)

        # Each DME encodes discriminative differences at each scale
        self.dme_skip1 = DME1(12, out_channels=24)  # 12 -> 24
        self.dme_skip2 = DME1(24, out_channels=48)  # 24 -> 48
        self.dme_skip3 = DME1(48, out_channels=96)  # 48 -> 96
        self.dme_skip4 = DME1(96, out_channels=192)  # 96 -> 192

        self.aemf = AEMF(in_channels=96, reduction=8)  # Output: 96 channels

        self.dec1 = DDSCR(288, 96)  # 96 (ECMF) + 192 (dme_skip4) = 288
        self.upconv1 = nn.ConvTranspose3d(96, 48, kernel_size=2, stride=2)

        self.dec2 = DDSCR(144, 48)  # 48 (upconv1) + 96 (dme_skip3) = 144
        self.upconv2 = nn.ConvTranspose3d(48, 24, kernel_size=2, stride=2)

        self.dec3 = DDSCR(72, 24)  # 24 (upconv2) + 48 (dme_skip2) = 72
        self.upconv3 = nn.ConvTranspose3d(24, 12, kernel_size=2, stride=2)

        self.dec4 = DDSCR(36, 12)  # 12 (upconv3) + 24 (dme_skip1) = 36

        self.final = nn.Conv3d(12, num_classes, kernel_size=1)

    def forward(self, t1, t1ce, t2, flair):

        # T1 pathway
        e1_t1 = self.enc1_t1(t1)
        d1_t1 = self.pool(e1_t1)

        e2_t1 = self.enc2_t1(d1_t1)
        d2_t1 = self.pool(e2_t1)

        e3_t1 = self.enc3_t1(d2_t1)
        d3_t1 = self.pool(e3_t1)

        e4_t1 = self.enc4_t1(d3_t1)

        # T1CE pathway
        e1_t1ce = self.enc1_t1ce(t1ce)
        d1_t1ce = self.pool(e1_t1ce)
        e2_t1ce = self.enc2_t1ce(d1_t1ce)
        d2_t1ce = self.pool(e2_t1ce)
        e3_t1ce = self.enc3_t1ce(d2_t1ce)
        d3_t1ce = self.pool(e3_t1ce)
        e4_t1ce = self.enc4_t1ce(d3_t1ce)

        # T2 pathway
        e1_t2 = self.enc1_t2(t2)
        d1_t2 = self.pool(e1_t2)
        e2_t2 = self.enc2_t2(d1_t2)
        d2_t2 = self.pool(e2_t2)
        e3_t2 = self.enc3_t2(d2_t2)
        d3_t2 = self.pool(e3_t2)
        e4_t2 = self.enc4_t2(d3_t2)

        # FLAIR pathway
        e1_flair = self.enc1_flair(flair)
        d1_flair = self.pool(e1_flair)
        e2_flair = self.enc2_flair(d1_flair)
        d2_flair = self.pool(e2_flair)
        e3_flair = self.enc3_flair(d2_flair)
        d3_flair = self.pool(e3_flair)
        e4_flair = self.enc4_flair(d3_flair)

        # DME encodes discriminative inter-modal boundaries
        dme_e1 = self.dme_skip1(e1_t1, e1_t1ce, e1_t2, e1_flair)  # [B, 24, D, H, W]
        dme_e2 = self.dme_skip2(e2_t1, e2_t1ce, e2_t2, e2_flair)  # [B, 48, D/2, H/2, W/2]
        dme_e3 = self.dme_skip3(e3_t1, e3_t1ce, e3_t2, e3_flair)  # [B, 96, D/4, H/4, W/4]
        dme_e4 = self.dme_skip4(e4_t1, e4_t1ce, e4_t2, e4_flair)  # [B, 192, D/8, H/8, W/8]

        bottleneck = self.aemf(e4_t1, e4_t1ce, e4_t2, e4_flair)  # [B, 96, D/8, H/8, W/8]

        # Level 1: Bottleneck + DME skip 4
        merge5 = torch.cat([bottleneck, dme_e4], dim=1)  # 96 + 192 = 288
        c5 = self.dec1(merge5)  # [B, 96, D/8, H/8, W/8]

        # Level 2: Upsample + DME skip 3
        up6 = self.upconv1(c5)  # [B, 48, D/4, H/4, W/4]
        merge6 = torch.cat([up6, dme_e3], dim=1)  # 48 + 96 = 144
        c6 = self.dec2(merge6)  # [B, 48, D/4, H/4, W/4]

        # Level 3: Upsample + DME skip 2
        up7 = self.upconv2(c6)  # [B, 24, D/2, H/2, W/2]
        merge7 = torch.cat([up7, dme_e2], dim=1)  # 24 + 48 = 72
        c7 = self.dec3(merge7)  # [B, 24, D/2, H/2, W/2]

        # Level 4: Upsample + DME skip 1
        up8 = self.upconv3(c7)  # [B, 12, D, H, W]
        merge8 = torch.cat([up8, dme_e1], dim=1)  # 12 + 24 = 36
        c8 = self.dec4(merge8)  # [B, 12, D, H, W]

        out = self.final(c8)
        return out

class DENet_s(nn.Module):

    def __init__(self, img_h, img_w, img_d, in_channels=1, num_classes=4):
        super().__init__()

        # Store original input size
        self.img_d = img_d
        self.img_h = img_h
        self.img_w = img_w

        # Max pooling layer
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder blocks for T1
        self.enc1_t1 = DDSCR(in_channels, 8)
        self.enc2_t1 = DDSCR(8, 16)
        self.enc3_t1 = DDSCR(16, 32)
        self.enc4_t1 = DDSCR(32, 64)

        # Encoder blocks for T1CE
        self.enc1_t1ce = DDSCR(in_channels, 8)
        self.enc2_t1ce = DDSCR(8, 16)
        self.enc3_t1ce = DDSCR(16, 32)
        self.enc4_t1ce = DDSCR(32, 64)

        # Encoder blocks for T2
        self.enc1_t2 = DDSCR(in_channels, 8)
        self.enc2_t2 = DDSCR(8, 16)
        self.enc3_t2 = DDSCR(16, 32)
        self.enc4_t2 = DDSCR(32, 64)

        # Encoder blocks for FLAIR
        self.enc1_flair = DDSCR(in_channels, 8)
        self.enc2_flair = DDSCR(8, 16)
        self.enc3_flair = DDSCR(16, 32)
        self.enc4_flair = DDSCR(32, 64)

        # Each DME encodes discriminative differences at each scale
        self.dme_skip1 = DME1(8, out_channels=16)    # 8 -> 16
        self.dme_skip2 = DME1(16, out_channels=32)   # 16 -> 32
        self.dme_skip3 = DME1(32, out_channels=64)   # 32 -> 64
        self.dme_skip4 = DME1(64, out_channels=128)  # 64 -> 128

        self.aemf = AEMF(in_channels=64, reduction=8)  # Output: 64 channels

        self.dec1 = DDSCR(192, 64)  # 64 (AEMF) + 128 (dme_skip4) = 192
        self.upconv1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)

        self.dec2 = DDSCR(96, 32)   # 32 (upconv1) + 64 (dme_skip3) = 96
        self.upconv2 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)

        self.dec3 = DDSCR(48, 16)   # 16 (upconv2) + 32 (dme_skip2) = 48
        self.upconv3 = nn.ConvTranspose3d(16, 8, kernel_size=2, stride=2)

        self.dec4 = DDSCR(24, 8)    # 8 (upconv3) + 16 (dme_skip1) = 24

        self.fusion = PGAM(
            feature_channels=[8, 16, 32, 64],  # From finest to coarsest
            output_size=(img_d, img_h, img_w),
            num_classes=num_classes,
            intermediate_channels=8  # Matches first layer
        )

    def forward(self, t1, t1ce, t2, flair):

        # T1 pathway
        e1_t1 = self.enc1_t1(t1)
        d1_t1 = self.pool(e1_t1)

        e2_t1 = self.enc2_t1(d1_t1)
        d2_t1 = self.pool(e2_t1)

        e3_t1 = self.enc3_t1(d2_t1)
        d3_t1 = self.pool(e3_t1)

        e4_t1 = self.enc4_t1(d3_t1)

        # T1CE pathway
        e1_t1ce = self.enc1_t1ce(t1ce)
        d1_t1ce = self.pool(e1_t1ce)
        e2_t1ce = self.enc2_t1ce(d1_t1ce)
        d2_t1ce = self.pool(e2_t1ce)
        e3_t1ce = self.enc3_t1ce(d2_t1ce)
        d3_t1ce = self.pool(e3_t1ce)
        e4_t1ce = self.enc4_t1ce(d3_t1ce)

        # T2 pathway
        e1_t2 = self.enc1_t2(t2)
        d1_t2 = self.pool(e1_t2)
        e2_t2 = self.enc2_t2(d1_t2)
        d2_t2 = self.pool(e2_t2)
        e3_t2 = self.enc3_t2(d2_t2)
        d3_t2 = self.pool(e3_t2)
        e4_t2 = self.enc4_t2(d3_t2)

        # FLAIR pathway
        e1_flair = self.enc1_flair(flair)
        d1_flair = self.pool(e1_flair)
        e2_flair = self.enc2_flair(d1_flair)
        d2_flair = self.pool(e2_flair)
        e3_flair = self.enc3_flair(d2_flair)
        d3_flair = self.pool(e3_flair)
        e4_flair = self.enc4_flair(d3_flair)

        # DME encodes discriminative inter-modal boundaries
        dme_e1 = self.dme_skip1(e1_t1, e1_t1ce, e1_t2, e1_flair)  # [B, 16, D, H, W]
        dme_e2 = self.dme_skip2(e2_t1, e2_t1ce, e2_t2, e2_flair)  # [B, 32, D/2, H/2, W/2]
        dme_e3 = self.dme_skip3(e3_t1, e3_t1ce, e3_t2, e3_flair)  # [B, 64, D/4, H/4, W/4]
        dme_e4 = self.dme_skip4(e4_t1, e4_t1ce, e4_t2, e4_flair)  # [B, 128, D/8, H/8, W/8]

        bottleneck = self.aemf(e4_t1, e4_t1ce, e4_t2, e4_flair)  # [B, 64, D/8, H/8, W/8]

        # Level 1: Bottleneck + DME skip 4
        merge5 = torch.cat([bottleneck, dme_e4], dim=1)  # 64 + 128 = 192
        c5 = self.dec1(merge5)  # [B, 64, D/8, H/8, W/8]

        # Level 2: Upsample + DME skip 3
        up6 = self.upconv1(c5)  # [B, 32, D/4, H/4, W/4]
        merge6 = torch.cat([up6, dme_e3], dim=1)  # 32 + 64 = 96
        c6 = self.dec2(merge6)  # [B, 32, D/4, H/4, W/4]

        # Level 3: Upsample + DME skip 2
        up7 = self.upconv2(c6)  # [B, 16, D/2, H/2, W/2]
        merge7 = torch.cat([up7, dme_e2], dim=1)  # 16 + 32 = 48
        c7 = self.dec3(merge7)  # [B, 16, D/2, H/2, W/2]

        # Level 4: Upsample + DME skip 1
        up8 = self.upconv3(c7)  # [B, 8, D, H, W]
        merge8 = torch.cat([up8, dme_e1], dim=1)  # 8 + 16 = 24
        c8 = self.dec4(merge8)  # [B, 8, D, H, W]

        # Pass features from finest to coarsest: [c8, c7, c6, c5]
        output = self.fusion([c8, c7, c6, c5])
        return output

class DENet_l(nn.Module):

    def __init__(self, img_h, img_w, img_d, in_channels=1, num_classes=4):
        super().__init__()

        # Store original input size
        self.img_d = img_d
        self.img_h = img_h
        self.img_w = img_w

        # Max pooling layer
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder blocks for T1
        self.enc1_t1 = DDSCR(in_channels, 16)
        self.enc2_t1 = DDSCR(16, 32)
        self.enc3_t1 = DDSCR(32, 64)
        self.enc4_t1 = DDSCR(64, 128)

        # Encoder blocks for T1CE
        self.enc1_t1ce = DDSCR(in_channels, 16)
        self.enc2_t1ce = DDSCR(16, 32)
        self.enc3_t1ce = DDSCR(32, 64)
        self.enc4_t1ce = DDSCR(64, 128)

        # Encoder blocks for T2
        self.enc1_t2 = DDSCR(in_channels, 16)
        self.enc2_t2 = DDSCR(16, 32)
        self.enc3_t2 = DDSCR(32, 64)
        self.enc4_t2 = DDSCR(64, 128)

        # Encoder blocks for FLAIR
        self.enc1_flair = DDSCR(in_channels, 16)
        self.enc2_flair = DDSCR(16, 32)
        self.enc3_flair = DDSCR(32, 64)
        self.enc4_flair = DDSCR(64, 128)

        # Each DME encodes discriminative differences at each scale
        self.dme_skip1 = DME1(16, out_channels=32)   # 16 -> 32
        self.dme_skip2 = DME1(32, out_channels=64)   # 32 -> 64
        self.dme_skip3 = DME1(64, out_channels=128)  # 64 -> 128
        self.dme_skip4 = DME1(128, out_channels=256) # 128 -> 256

        self.aemf = AEMF(in_channels=128, reduction=8)  # Output: 128 channels

        self.dec1 = DDSCR(384, 128)  # 128 (AEMF) + 256 (dme_skip4) = 384
        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)

        self.dec2 = DDSCR(192, 64)   # 64 (upconv1) + 128 (dme_skip3) = 192
        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)

        self.dec3 = DDSCR(96, 32)    # 32 (upconv2) + 64 (dme_skip2) = 96
        self.upconv3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)

        self.dec4 = DDSCR(48, 16)    # 16 (upconv3) + 32 (dme_skip1) = 48

        self.fusion = PGAM(
            feature_channels=[16, 32, 64, 128],  # From finest to coarsest
            output_size=(img_d, img_h, img_w),
            num_classes=num_classes,
            intermediate_channels=16  # Matches first layer
        )

    def forward(self, t1, t1ce, t2, flair):

        # T1 pathway
        e1_t1 = self.enc1_t1(t1)
        d1_t1 = self.pool(e1_t1)

        e2_t1 = self.enc2_t1(d1_t1)
        d2_t1 = self.pool(e2_t1)

        e3_t1 = self.enc3_t1(d2_t1)
        d3_t1 = self.pool(e3_t1)

        e4_t1 = self.enc4_t1(d3_t1)

        # T1CE pathway
        e1_t1ce = self.enc1_t1ce(t1ce)
        d1_t1ce = self.pool(e1_t1ce)
        e2_t1ce = self.enc2_t1ce(d1_t1ce)
        d2_t1ce = self.pool(e2_t1ce)
        e3_t1ce = self.enc3_t1ce(d2_t1ce)
        d3_t1ce = self.pool(e3_t1ce)
        e4_t1ce = self.enc4_t1ce(d3_t1ce)

        # T2 pathway
        e1_t2 = self.enc1_t2(t2)
        d1_t2 = self.pool(e1_t2)
        e2_t2 = self.enc2_t2(d1_t2)
        d2_t2 = self.pool(e2_t2)
        e3_t2 = self.enc3_t2(d2_t2)
        d3_t2 = self.pool(e3_t2)
        e4_t2 = self.enc4_t2(d3_t2)

        # FLAIR pathway
        e1_flair = self.enc1_flair(flair)
        d1_flair = self.pool(e1_flair)
        e2_flair = self.enc2_flair(d1_flair)
        d2_flair = self.pool(e2_flair)
        e3_flair = self.enc3_flair(d2_flair)
        d3_flair = self.pool(e3_flair)
        e4_flair = self.enc4_flair(d3_flair)

        # DME encodes discriminative inter-modal boundaries
        dme_e1 = self.dme_skip1(e1_t1, e1_t1ce, e1_t2, e1_flair)  # [B, 32, D, H, W]
        dme_e2 = self.dme_skip2(e2_t1, e2_t1ce, e2_t2, e2_flair)  # [B, 64, D/2, H/2, W/2]
        dme_e3 = self.dme_skip3(e3_t1, e3_t1ce, e3_t2, e3_flair)  # [B, 128, D/4, H/4, W/4]
        dme_e4 = self.dme_skip4(e4_t1, e4_t1ce, e4_t2, e4_flair)  # [B, 256, D/8, H/8, W/8]

        bottleneck = self.aemf(e4_t1, e4_t1ce, e4_t2, e4_flair)  # [B, 128, D/8, H/8, W/8]

        # Level 1: Bottleneck + DME skip 4
        merge5 = torch.cat([bottleneck, dme_e4], dim=1)  # 128 + 256 = 384
        c5 = self.dec1(merge5)  # [B, 128, D/8, H/8, W/8]

        # Level 2: Upsample + DME skip 3
        up6 = self.upconv1(c5)  # [B, 64, D/4, H/4, W/4]
        merge6 = torch.cat([up6, dme_e3], dim=1)  # 64 + 128 = 192
        c6 = self.dec2(merge6)  # [B, 64, D/4, H/4, W/4]

        # Level 3: Upsample + DME skip 2
        up7 = self.upconv2(c6)  # [B, 32, D/2, H/2, W/2]
        merge7 = torch.cat([up7, dme_e2], dim=1)  # 32 + 64 = 96
        c7 = self.dec3(merge7)  # [B, 32, D/2, H/2, W/2]

        # Level 4: Upsample + DME skip 1
        up8 = self.upconv3(c7)  # [B, 16, D, H, W]
        merge8 = torch.cat([up8, dme_e1], dim=1)  # 16 + 32 = 48
        c8 = self.dec4(merge8)  # [B, 16, D, H, W]

        # Pass features from finest to coarsest: [c8, c7, c6, c5]
        output = self.fusion([c8, c7, c6, c5])
        return output

