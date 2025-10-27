import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class DepthwiseChannelAttention(nn.Module):

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.depthwise = nn.Conv3d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False
        )

        self.pointwise = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.norm = nn.InstanceNorm3d(channels, affine=True)
        self.act = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depthwise(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.pointwise(out)
        gap = F.adaptive_avg_pool3d(out, 1)
        weights = self.sigmoid(gap)
        return weights



class CrossScaleGatingBlock(nn.Module):
    """Cross-gating between two feature maps (fine and coarse scales)."""

    def __init__(self, channels: int):
        super().__init__()

        self.gate_conv = nn.Sequential(
            nn.Conv3d(2 * channels, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.GELU()
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, fine: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        concatenated = torch.cat([fine, coarse], dim=1)
        gate = self.sigmoid(self.gate_conv(concatenated))
        gated_coarse = gate * coarse
        fused = fine + gated_coarse
        return fused


# ========================================
# IMPROVED: Fine → Coarse Progressive Gating (4 Decoder Stages Only)
# ========================================

class PGAM(nn.Module):
    """
    - Progressive Gating Attention Module (PGAM)
    - Uses only 4 decoder stages (no bottleneck)
    - Uses intermediate_channels (12) for richer feature representation
    - Final fusion reduces from intermediate_channels to num_classes
    """

    def __init__(
            self,
            feature_channels: List[int],
            output_size: Tuple[int, int, int],
            num_classes: int = 4,
            intermediate_channels: int = 12
    ):
        super().__init__()

        assert len(feature_channels) >= 2, "Need at least 2 scales"

        self.output_size = output_size
        self.num_scales = len(feature_channels)
        self.num_classes = num_classes
        self.intermediate_channels = intermediate_channels

        # Channel alignment to intermediate dimension (12 channels)
        self.channel_align = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(ch, intermediate_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(intermediate_channels),
                nn.GELU()
            )
            for ch in feature_channels
        ])

        # Progressive cross-gating on intermediate channels
        self.cross_gates = nn.ModuleList([
            CrossScaleGatingBlock(intermediate_channels)
            for _ in range(self.num_scales - 1)
        ])

        # Channel attention only (removed spatial attention for efficiency)
        self.channel_attention = DepthwiseChannelAttention(
            channels=intermediate_channels,
            kernel_size=3
        )

        # Final fusion: intermediate_channels → num_classes
        self.final_fusion = nn.Sequential(
            nn.Conv3d(intermediate_channels, intermediate_channels * 2,
                      kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(intermediate_channels * 2),
            nn.GELU(),
            nn.Dropout3d(0.1),
            nn.Conv3d(intermediate_channels * 2, num_classes,
                      kernel_size=1, bias=True)
        )

    def forward(self, feature_list: List[torch.Tensor]) -> torch.Tensor:
        """Fine → Coarse progressive gating with intermediate channels"""
        assert len(feature_list) == self.num_scales

        # Channel alignment to intermediate dimension
        aligned_features = []
        for i, feature in enumerate(feature_list):
            aligned = self.channel_align[i](feature)
            aligned_features.append(aligned)

        # Upsample to target size
        upsampled_features = []
        for aligned in aligned_features:
            upsampled = F.interpolate(
                aligned,
                size=self.output_size,
                mode='trilinear',
                align_corners=False
            )
            upsampled_features.append(upsampled)

        # Progressive gating: Fine → Coarse
        fused = upsampled_features[0]  # Start with finest

        for i in range(1, self.num_scales):
            coarse_feature = upsampled_features[i]
            gate_module = self.cross_gates[i - 1]
            fused = gate_module(fused, coarse_feature)

        # Channel attention only
        channel_weights = self.channel_attention(fused)
        fused = fused * channel_weights

        # Final convolution: intermediate_channels → num_classes
        output = self.final_fusion(fused)

        return output


# ========================================
# ABLATION 1: Coarse → Fine Progressive Gating
# ========================================

class PGAM1(nn.Module):
    """
    Ablation 1: Coarse to Fine Progressive Gating with Intermediate Channels
    Starts from coarsest scale, progressively gates in finer scales
    """

    def __init__(
            self,
            feature_channels: List[int],
            output_size: Tuple[int, int, int],
            num_classes: int = 4,
            intermediate_channels: int = 12
    ):
        super().__init__()

        assert len(feature_channels) >= 2, "Need at least 2 scales"

        self.output_size = output_size
        self.num_scales = len(feature_channels)
        self.num_classes = num_classes
        self.intermediate_channels = intermediate_channels

        # Channel alignment to intermediate dimension
        self.channel_align = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(ch, intermediate_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(intermediate_channels),
                nn.GELU()
            )
            for ch in feature_channels
        ])

        # Progressive cross-gating
        self.cross_gates = nn.ModuleList([
            CrossScaleGatingBlock(intermediate_channels)
            for _ in range(self.num_scales - 1)
        ])

        # Channel attention only
        self.channel_attention = DepthwiseChannelAttention(
            channels=intermediate_channels,
            kernel_size=3
        )

        # Final fusion
        self.final_fusion = nn.Sequential(
            nn.Conv3d(intermediate_channels, intermediate_channels * 2,
                      kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(intermediate_channels * 2),
            nn.GELU(),
            nn.Dropout3d(0.1),
            nn.Conv3d(intermediate_channels * 2, num_classes,
                      kernel_size=1, bias=True)
        )

    def forward(self, feature_list: List[torch.Tensor]) -> torch.Tensor:
        """Coarse → Fine progressive gating"""
        assert len(feature_list) == self.num_scales

        # Channel alignment
        aligned_features = []
        for i, feature in enumerate(feature_list):
            aligned = self.channel_align[i](feature)
            aligned_features.append(aligned)

        # Upsample to target size
        upsampled_features = []
        for aligned in aligned_features:
            upsampled = F.interpolate(
                aligned,
                size=self.output_size,
                mode='trilinear',
                align_corners=False
            )
            upsampled_features.append(upsampled)

        # Progressive gating: Coarse → Fine (REVERSED)
        fused = upsampled_features[-1]  # Start with coarsest

        for i in range(self.num_scales - 2, -1, -1):
            fine_feature = upsampled_features[i]
            gate_module = self.cross_gates[i]
            fused = gate_module(fused, fine_feature)

        # Channel attention only
        channel_weights = self.channel_attention(fused)
        fused = fused * channel_weights

        # Final convolution
        output = self.final_fusion(fused)

        return output


# ========================================
# ABLATION 2: Parallel Gating Fusion
# ========================================

class PGAM2(nn.Module):
    """
    Ablation 2: Parallel Gating Fusion with Intermediate Channels
    All scales are gated independently, then fused together
    """

    def __init__(
            self,
            feature_channels: List[int],
            output_size: Tuple[int, int, int],
            num_classes: int = 4,
            intermediate_channels: int = 12
    ):
        super().__init__()

        assert len(feature_channels) >= 2, "Need at least 2 scales"

        self.output_size = output_size
        self.num_scales = len(feature_channels)
        self.num_classes = num_classes
        self.intermediate_channels = intermediate_channels

        # Channel alignment to intermediate dimension
        self.channel_align = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(ch, intermediate_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(intermediate_channels),
                nn.GELU()
            )
            for ch in feature_channels
        ])

        # Individual gate for each scale (independent gating)
        self.scale_gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(intermediate_channels, intermediate_channels,
                          kernel_size=1, bias=False),
                nn.InstanceNorm3d(intermediate_channels),
                nn.GELU(),
                nn.Conv3d(intermediate_channels, intermediate_channels,
                          kernel_size=1, bias=False),
                nn.Sigmoid()
            )
            for _ in range(self.num_scales)
        ])

        # Learnable scale importance weights
        self.scale_weights = nn.Parameter(torch.ones(self.num_scales))

        # Channel attention only
        self.channel_attention = DepthwiseChannelAttention(
            channels=intermediate_channels,
            kernel_size=3
        )

        # Final fusion
        self.final_fusion = nn.Sequential(
            nn.Conv3d(intermediate_channels, intermediate_channels * 2,
                      kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(intermediate_channels * 2),
            nn.GELU(),
            nn.Dropout3d(0.1),
            nn.Conv3d(intermediate_channels * 2, num_classes,
                      kernel_size=1, bias=True)
        )

    def forward(self, feature_list: List[torch.Tensor]) -> torch.Tensor:
        """Parallel independent gating then weighted fusion"""
        assert len(feature_list) == self.num_scales

        # Channel alignment
        aligned_features = []
        for i, feature in enumerate(feature_list):
            aligned = self.channel_align[i](feature)
            aligned_features.append(aligned)

        # Upsample to target size
        upsampled_features = []
        for aligned in aligned_features:
            upsampled = F.interpolate(
                aligned,
                size=self.output_size,
                mode='trilinear',
                align_corners=False
            )
            upsampled_features.append(upsampled)

        # Parallel gating: Each scale independently gated
        gated_features = []
        for i, feature in enumerate(upsampled_features):
            gate = self.scale_gates[i](feature)
            gated = feature * gate
            gated_features.append(gated)

        # Normalize scale weights with softmax
        normalized_weights = F.softmax(self.scale_weights, dim=0)

        # Weighted sum fusion
        fused = torch.zeros_like(gated_features[0])
        for i, gated in enumerate(gated_features):
            fused = fused + normalized_weights[i] * gated

        # Channel attention only
        channel_weights = self.channel_attention(fused)
        fused = fused * channel_weights

        # Final convolution
        output = self.final_fusion(fused)

        return output



