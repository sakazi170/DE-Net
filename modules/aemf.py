import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SelfAttention(nn.Module):
    """Standard self-attention without chunking."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.channels = channels

        self.to_q = nn.Conv3d(channels, channels // reduction, kernel_size=1, bias=False)
        self.to_k = nn.Conv3d(channels, channels // reduction, kernel_size=1, bias=False)
        self.to_v = nn.Conv3d(channels, channels, kernel_size=1, bias=False)

        self.proj = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.InstanceNorm3d(channels)

        self.scale = (channels // reduction) ** -0.5

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, D, H, W = x.shape
        N = D * H * W

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q_flat = q.flatten(2)
        k_flat = k.flatten(2)
        v_flat = v.flatten(2)

        attn = torch.bmm(q_flat.transpose(1, 2), k_flat) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(v_flat, attn.transpose(1, 2))
        out = out.reshape(B, C, D, H, W)

        out = self.proj(out)
        out = self.norm(out)
        attended = out + x
        return q, k, v, attended


class CrossAttention(nn.Module):
    """Standard cross-attention without chunking."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.channels = channels

        self.to_q = nn.Conv3d(channels, channels // reduction, kernel_size=1, bias=False)
        self.to_k = nn.Conv3d(channels, channels // reduction, kernel_size=1, bias=False)
        self.to_v = nn.Conv3d(channels, channels, kernel_size=1, bias=False)

        self.proj = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.InstanceNorm3d(channels)

        self.scale = (channels // reduction) ** -0.5

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x1.shape

        q = self.to_q(x1)
        k = self.to_k(x2)
        v = self.to_v(x2)

        q_flat = q.flatten(2)
        k_flat = k.flatten(2)
        v_flat = v.flatten(2)

        attn = torch.bmm(q_flat.transpose(1, 2), k_flat) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(v_flat, attn.transpose(1, 2))
        out = out.reshape(B, C, D, H, W)

        out = self.proj(out)
        out = self.norm(out)
        out = out + x1
        return out


class AdaptiveEntropy(nn.Module):
    """Adaptive spatial-aware entropy calculation"""

    def __init__(self, channels: int, bins: int = 128, reduction: int = 8):
        super().__init__()
        self.bins = bins

        self.context_encoder = nn.Sequential(
            nn.AdaptiveAvgPool3d(4),
            nn.Conv3d(channels, max(1, channels // reduction), 1, bias=False),
            nn.InstanceNorm3d(max(1, channels // reduction)),
            nn.GELU(),
            nn.Conv3d(max(1, channels // reduction), 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape

        spatial_weights = self.context_encoder(x)
        spatial_weights = F.interpolate(spatial_weights, size=(D, H, W),
                                        mode='trilinear', align_corners=False)

        weighted_x = x * spatial_weights

        x_min, x_max = weighted_x.min(), weighted_x.max()
        x_norm = (weighted_x - x_min) / (x_max - x_min + 1e-8)

        x_flat = x_norm.flatten()
        hist = torch.histc(x_flat, bins=self.bins, min=0.0, max=1.0)
        prob = hist / (hist.sum() + 1e-10)
        entropy = -torch.sum(prob * torch.log2(prob + 1e-10))
        return entropy


# ========================================
# ORIGINAL: PAIRING (T1-T1CE) and (T2-FLAIR) - WITH ENTROPY
# Baseline configuration from the original module
# ========================================

class AEMF(nn.Module):
    """
    Adaptive Entropy-based Modality Fusion (AEMF) module
    Original Pairing Strategy: (T1 ↔ T1CE) and (T2 ↔ FLAIR) WITH Entropy
    This is the baseline configuration for comparison
    """

    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()

        self.in_channels = in_channels
        self.reduced_channels = in_channels // reduction

        # Self-attention for each modality
        self.self_attn_t1 = SelfAttention(in_channels, reduction)
        self.self_attn_t1ce = SelfAttention(in_channels, reduction)
        self.self_attn_t2 = SelfAttention(in_channels, reduction)
        self.self_attn_flair = SelfAttention(in_channels, reduction)

        # Entropy calculators
        self.entropy_calc_full = AdaptiveEntropy(in_channels, bins=128, reduction=reduction)
        self.entropy_calc_reduced = AdaptiveEntropy(self.reduced_channels, bins=128,
                                                        reduction=max(1, reduction // 2))

        # Cross-attention: T1 ↔ T1CE (original pair 1)
        self.cross_attn_t1_to_t1ce = CrossAttention(in_channels, reduction)
        self.cross_attn_t1ce_to_t1 = CrossAttention(in_channels, reduction)

        # Cross-attention: T2 ↔ FLAIR (original pair 2)
        self.cross_attn_t2_to_flair = CrossAttention(in_channels, reduction)
        self.cross_attn_flair_to_t2 = CrossAttention(in_channels, reduction)

        # Fusion layers
        self.fuse_pair1 = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(in_channels),
            nn.GELU()
        )

        self.fuse_pair2 = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(in_channels),
            nn.GELU()
        )

        # Final aggregation
        self.final_fusion = nn.Sequential(
            nn.Conv3d(in_channels * 6, in_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(in_channels * 2),
            nn.GELU(),
            nn.Dropout3d(0.1),
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(in_channels),
            nn.GELU()
        )

    def forward(self, t1: torch.Tensor, t1ce: torch.Tensor,
                t2: torch.Tensor, flair: torch.Tensor) -> torch.Tensor:
        # Stage 1: Self-Attention for each modality
        q_t1, k_t1, v_t1, attended_t1 = self.self_attn_t1(t1)
        q_t1ce, k_t1ce, v_t1ce, attended_t1ce = self.self_attn_t1ce(t1ce)
        q_t2, k_t2, v_t2, attended_t2 = self.self_attn_t2(t2)
        q_flair, k_flair, v_flair, attended_flair = self.self_attn_flair(flair)

        # Stage 2: Entropy Calculation
        # For T1
        e_q_t1 = self.entropy_calc_reduced(q_t1)
        e_k_t1 = self.entropy_calc_reduced(k_t1)
        e_v_t1 = self.entropy_calc_full(v_t1)
        entropy_t1 = (e_q_t1 + e_k_t1 + e_v_t1) / 3.0

        # For T1CE
        e_q_t1ce = self.entropy_calc_reduced(q_t1ce)
        e_k_t1ce = self.entropy_calc_reduced(k_t1ce)
        e_v_t1ce = self.entropy_calc_full(v_t1ce)
        entropy_t1ce = (e_q_t1ce + e_k_t1ce + e_v_t1ce) / 3.0

        # For T2
        e_q_t2 = self.entropy_calc_reduced(q_t2)
        e_k_t2 = self.entropy_calc_reduced(k_t2)
        e_v_t2 = self.entropy_calc_full(v_t2)
        entropy_t2 = (e_q_t2 + e_k_t2 + e_v_t2) / 3.0

        # For FLAIR
        e_q_flair = self.entropy_calc_reduced(q_flair)
        e_k_flair = self.entropy_calc_reduced(k_flair)
        e_v_flair = self.entropy_calc_full(v_flair)
        entropy_flair = (e_q_flair + e_k_flair + e_v_flair) / 3.0

        # Compute entropy weights
        entropy_stack = torch.stack([entropy_t1, entropy_t1ce, entropy_t2, entropy_flair])
        entropy_weights = F.softmax(entropy_stack, dim=0)

        # Stage 3: Cross-Modal Attention - Original Pairs
        # Pair 1: T1 ↔ T1CE
        cross_t1_to_t1ce = self.cross_attn_t1_to_t1ce(attended_t1, attended_t1ce)
        cross_t1ce_to_t1 = self.cross_attn_t1ce_to_t1(attended_t1ce, attended_t1)
        cross_pair1 = self.fuse_pair1(torch.cat([cross_t1_to_t1ce, cross_t1ce_to_t1], dim=1))

        # Pair 2: T2 ↔ FLAIR
        cross_t2_to_flair = self.cross_attn_t2_to_flair(attended_t2, attended_flair)
        cross_flair_to_t2 = self.cross_attn_flair_to_t2(attended_flair, attended_t2)
        cross_pair2 = self.fuse_pair2(torch.cat([cross_t2_to_flair, cross_flair_to_t2], dim=1))

        # Stage 4: Entropy-Weighted Fusion
        weighted_t1 = entropy_weights[0] * attended_t1
        weighted_t1ce = entropy_weights[1] * attended_t1ce
        weighted_t2 = entropy_weights[2] * attended_t2
        weighted_flair = entropy_weights[3] * attended_flair

        # Stage 5: Final Aggregation
        all_features = torch.cat([
            weighted_t1,
            weighted_t1ce,
            weighted_t2,
            weighted_flair,
            cross_pair1,
            cross_pair2
        ], dim=1)

        output = self.final_fusion(all_features)
        return output


# ========================================
# ABLATION 1: PAIRING (T1-T2) and (T1CE-FLAIR)
# Rationale: Non-contrast vs Contrast-enhanced pairs
# ========================================

class AEMF1(nn.Module):
    """
    Pairing Strategy 1: (T1 ↔ T2) and (T1CE ↔ FLAIR)
    Groups non-contrast (T1, T2) and contrast-enhanced (T1CE, FLAIR) modalities
    """

    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()

        self.in_channels = in_channels
        self.reduced_channels = in_channels // reduction

        # Self-attention for each modality
        self.self_attn_t1 = SelfAttention(in_channels, reduction)
        self.self_attn_t1ce = SelfAttention(in_channels, reduction)
        self.self_attn_t2 = SelfAttention(in_channels, reduction)
        self.self_attn_flair = SelfAttention(in_channels, reduction)

        # Entropy calculators
        self.entropy_calc_full = AdaptiveEntropy(in_channels, bins=128, reduction=reduction)
        self.entropy_calc_reduced = AdaptiveEntropy(self.reduced_channels, bins=128,
                                                        reduction=max(1, reduction // 2))

        # Cross-attention: T1 ↔ T2
        self.cross_attn_t1_to_t2 = CrossAttention(in_channels, reduction)
        self.cross_attn_t2_to_t1 = CrossAttention(in_channels, reduction)

        # Cross-attention: T1CE ↔ FLAIR
        self.cross_attn_t1ce_to_flair = CrossAttention(in_channels, reduction)
        self.cross_attn_flair_to_t1ce = CrossAttention(in_channels, reduction)

        # Fusion layers
        self.fuse_pair1 = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(in_channels),
            nn.GELU()
        )

        self.fuse_pair2 = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(in_channels),
            nn.GELU()
        )

        # Final aggregation
        self.final_fusion = nn.Sequential(
            nn.Conv3d(in_channels * 6, in_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(in_channels * 2),
            nn.GELU(),
            nn.Dropout3d(0.1),
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(in_channels),
            nn.GELU()
        )

    def forward(self, t1: torch.Tensor, t1ce: torch.Tensor,
                t2: torch.Tensor, flair: torch.Tensor) -> torch.Tensor:
        # Self-Attention
        q_t1, k_t1, v_t1, attended_t1 = self.self_attn_t1(t1)
        q_t1ce, k_t1ce, v_t1ce, attended_t1ce = self.self_attn_t1ce(t1ce)
        q_t2, k_t2, v_t2, attended_t2 = self.self_attn_t2(t2)
        q_flair, k_flair, v_flair, attended_flair = self.self_attn_flair(flair)

        # Entropy Calculation
        e_q_t1 = self.entropy_calc_reduced(q_t1)
        e_k_t1 = self.entropy_calc_reduced(k_t1)
        e_v_t1 = self.entropy_calc_full(v_t1)
        entropy_t1 = (e_q_t1 + e_k_t1 + e_v_t1) / 3.0

        e_q_t1ce = self.entropy_calc_reduced(q_t1ce)
        e_k_t1ce = self.entropy_calc_reduced(k_t1ce)
        e_v_t1ce = self.entropy_calc_full(v_t1ce)
        entropy_t1ce = (e_q_t1ce + e_k_t1ce + e_v_t1ce) / 3.0

        e_q_t2 = self.entropy_calc_reduced(q_t2)
        e_k_t2 = self.entropy_calc_reduced(k_t2)
        e_v_t2 = self.entropy_calc_full(v_t2)
        entropy_t2 = (e_q_t2 + e_k_t2 + e_v_t2) / 3.0

        e_q_flair = self.entropy_calc_reduced(q_flair)
        e_k_flair = self.entropy_calc_reduced(k_flair)
        e_v_flair = self.entropy_calc_full(v_flair)
        entropy_flair = (e_q_flair + e_k_flair + e_v_flair) / 3.0

        # Entropy weights
        entropy_stack = torch.stack([entropy_t1, entropy_t1ce, entropy_t2, entropy_flair])
        entropy_weights = F.softmax(entropy_stack, dim=0)

        # Cross-Modal Attention: Pair 1 (T1 ↔ T2)
        cross_t1_to_t2 = self.cross_attn_t1_to_t2(attended_t1, attended_t2)
        cross_t2_to_t1 = self.cross_attn_t2_to_t1(attended_t2, attended_t1)
        cross_pair1 = self.fuse_pair1(torch.cat([cross_t1_to_t2, cross_t2_to_t1], dim=1))

        # Cross-Modal Attention: Pair 2 (T1CE ↔ FLAIR)
        cross_t1ce_to_flair = self.cross_attn_t1ce_to_flair(attended_t1ce, attended_flair)
        cross_flair_to_t1ce = self.cross_attn_flair_to_t1ce(attended_flair, attended_t1ce)
        cross_pair2 = self.fuse_pair2(torch.cat([cross_t1ce_to_flair, cross_flair_to_t1ce], dim=1))

        # Entropy-Weighted Fusion
        weighted_t1 = entropy_weights[0] * attended_t1
        weighted_t1ce = entropy_weights[1] * attended_t1ce
        weighted_t2 = entropy_weights[2] * attended_t2
        weighted_flair = entropy_weights[3] * attended_flair

        # Final Aggregation
        all_features = torch.cat([
            weighted_t1,
            weighted_t1ce,
            weighted_t2,
            weighted_flair,
            cross_pair1,
            cross_pair2
        ], dim=1)

        output = self.final_fusion(all_features)
        return output


# ========================================
# ABLATION 2: PAIRING (T1-FLAIR) and (T1CE-T2)
# Rationale: Anatomical detail (T1, FLAIR) vs Pathology detection (T1CE, T2)
# ========================================

class AEMF2(nn.Module):
    """
    Pairing Strategy 2: (T1 ↔ FLAIR) and (T1CE ↔ T2)
    Groups anatomical structure (T1, FLAIR) and pathology-sensitive (T1CE, T2) modalities
    """

    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()

        self.in_channels = in_channels
        self.reduced_channels = in_channels // reduction

        # Self-attention for each modality
        self.self_attn_t1 = SelfAttention(in_channels, reduction)
        self.self_attn_t1ce = SelfAttention(in_channels, reduction)
        self.self_attn_t2 = SelfAttention(in_channels, reduction)
        self.self_attn_flair = SelfAttention(in_channels, reduction)

        # Entropy calculators
        self.entropy_calc_full = AdaptiveEntropy(in_channels, bins=128, reduction=reduction)
        self.entropy_calc_reduced = AdaptiveEntropy(self.reduced_channels, bins=128,
                                                        reduction=max(1, reduction // 2))

        # Cross-attention: T1 ↔ FLAIR
        self.cross_attn_t1_to_flair = CrossAttention(in_channels, reduction)
        self.cross_attn_flair_to_t1 = CrossAttention(in_channels, reduction)

        # Cross-attention: T1CE ↔ T2
        self.cross_attn_t1ce_to_t2 = CrossAttention(in_channels, reduction)
        self.cross_attn_t2_to_t1ce = CrossAttention(in_channels, reduction)

        # Fusion layers
        self.fuse_pair1 = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(in_channels),
            nn.GELU()
        )

        self.fuse_pair2 = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(in_channels),
            nn.GELU()
        )

        # Final aggregation
        self.final_fusion = nn.Sequential(
            nn.Conv3d(in_channels * 6, in_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(in_channels * 2),
            nn.GELU(),
            nn.Dropout3d(0.1),
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(in_channels),
            nn.GELU()
        )

    def forward(self, t1: torch.Tensor, t1ce: torch.Tensor,
                t2: torch.Tensor, flair: torch.Tensor) -> torch.Tensor:
        # Self-Attention
        q_t1, k_t1, v_t1, attended_t1 = self.self_attn_t1(t1)
        q_t1ce, k_t1ce, v_t1ce, attended_t1ce = self.self_attn_t1ce(t1ce)
        q_t2, k_t2, v_t2, attended_t2 = self.self_attn_t2(t2)
        q_flair, k_flair, v_flair, attended_flair = self.self_attn_flair(flair)

        # Entropy Calculation
        e_q_t1 = self.entropy_calc_reduced(q_t1)
        e_k_t1 = self.entropy_calc_reduced(k_t1)
        e_v_t1 = self.entropy_calc_full(v_t1)
        entropy_t1 = (e_q_t1 + e_k_t1 + e_v_t1) / 3.0

        e_q_t1ce = self.entropy_calc_reduced(q_t1ce)
        e_k_t1ce = self.entropy_calc_reduced(k_t1ce)
        e_v_t1ce = self.entropy_calc_full(v_t1ce)
        entropy_t1ce = (e_q_t1ce + e_k_t1ce + e_v_t1ce) / 3.0

        e_q_t2 = self.entropy_calc_reduced(q_t2)
        e_k_t2 = self.entropy_calc_reduced(k_t2)
        e_v_t2 = self.entropy_calc_full(v_t2)
        entropy_t2 = (e_q_t2 + e_k_t2 + e_v_t2) / 3.0

        e_q_flair = self.entropy_calc_reduced(q_flair)
        e_k_flair = self.entropy_calc_reduced(k_flair)
        e_v_flair = self.entropy_calc_full(v_flair)
        entropy_flair = (e_q_flair + e_k_flair + e_v_flair) / 3.0

        # Entropy weights
        entropy_stack = torch.stack([entropy_t1, entropy_t1ce, entropy_t2, entropy_flair])
        entropy_weights = F.softmax(entropy_stack, dim=0)

        # Cross-Modal Attention: Pair 1 (T1 ↔ FLAIR)
        cross_t1_to_flair = self.cross_attn_t1_to_flair(attended_t1, attended_flair)
        cross_flair_to_t1 = self.cross_attn_flair_to_t1(attended_flair, attended_t1)
        cross_pair1 = self.fuse_pair1(torch.cat([cross_t1_to_flair, cross_flair_to_t1], dim=1))

        # Cross-Modal Attention: Pair 2 (T1CE ↔ T2)
        cross_t1ce_to_t2 = self.cross_attn_t1ce_to_t2(attended_t1ce, attended_t2)
        cross_t2_to_t1ce = self.cross_attn_t2_to_t1ce(attended_t2, attended_t1ce)
        cross_pair2 = self.fuse_pair2(torch.cat([cross_t1ce_to_t2, cross_t2_to_t1ce], dim=1))

        # Entropy-Weighted Fusion
        weighted_t1 = entropy_weights[0] * attended_t1
        weighted_t1ce = entropy_weights[1] * attended_t1ce
        weighted_t2 = entropy_weights[2] * attended_t2
        weighted_flair = entropy_weights[3] * attended_flair

        # Final Aggregation
        all_features = torch.cat([
            weighted_t1,
            weighted_t1ce,
            weighted_t2,
            weighted_flair,
            cross_pair1,
            cross_pair2
        ], dim=1)

        output = self.final_fusion(all_features)
        return output


# ========================================
# ABLATION 3: WITHOUT ENTROPY (Four Modalities)
# Tests the contribution of entropy-based weighting
# ========================================

class AEMF_NoEntropy(nn.Module):
    """
    Four-modality fusion WITHOUT entropy weighting
    Uses equal weights for all modalities
    """

    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()

        self.in_channels = in_channels

        # Self-attention for each modality
        self.self_attn_t1 = SelfAttention(in_channels, reduction)
        self.self_attn_t1ce = SelfAttention(in_channels, reduction)
        self.self_attn_t2 = SelfAttention(in_channels, reduction)
        self.self_attn_flair = SelfAttention(in_channels, reduction)

        # Cross-attention between modalities
        self.cross_attn_t1_to_t1ce = CrossAttention(in_channels, reduction)
        self.cross_attn_t1ce_to_t1 = CrossAttention(in_channels, reduction)
        self.cross_attn_t2_to_flair = CrossAttention(in_channels, reduction)
        self.cross_attn_flair_to_t2 = CrossAttention(in_channels, reduction)

        # Fusion layers
        self.fuse_pair1 = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(in_channels),
            nn.GELU()
        )

        self.fuse_pair2 = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(in_channels),
            nn.GELU()
        )

        # Final aggregation
        self.final_fusion = nn.Sequential(
            nn.Conv3d(in_channels * 6, in_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(in_channels * 2),
            nn.GELU(),
            nn.Dropout3d(0.1),
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(in_channels),
            nn.GELU()
        )

    def forward(self, t1: torch.Tensor, t1ce: torch.Tensor,
                t2: torch.Tensor, flair: torch.Tensor) -> torch.Tensor:
        # Self-Attention (only need attended outputs, not Q,K,V)
        _, _, _, attended_t1 = self.self_attn_t1(t1)
        _, _, _, attended_t1ce = self.self_attn_t1ce(t1ce)
        _, _, _, attended_t2 = self.self_attn_t2(t2)
        _, _, _, attended_flair = self.self_attn_flair(flair)

        # Cross-Modal Attention
        cross_t1_to_t1ce = self.cross_attn_t1_to_t1ce(attended_t1, attended_t1ce)
        cross_t1ce_to_t1 = self.cross_attn_t1ce_to_t1(attended_t1ce, attended_t1)
        cross_pair1 = self.fuse_pair1(torch.cat([cross_t1_to_t1ce, cross_t1ce_to_t1], dim=1))

        cross_t2_to_flair = self.cross_attn_t2_to_flair(attended_t2, attended_flair)
        cross_flair_to_t2 = self.cross_attn_flair_to_t2(attended_flair, attended_t2)
        cross_pair2 = self.fuse_pair2(torch.cat([cross_t2_to_flair, cross_flair_to_t2], dim=1))

        # Equal weighting (no entropy)
        # Simple concatenation without weighted fusion
        all_features = torch.cat([
            attended_t1,
            attended_t1ce,
            attended_t2,
            attended_flair,
            cross_pair1,
            cross_pair2
        ], dim=1)

        output = self.final_fusion(all_features)
        return output


