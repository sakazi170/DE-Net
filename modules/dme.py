import torch
import torch.nn as nn


class DME(nn.Module):
    """
    Differential Modality Encoding (DME) for Skip Connections

    Theoretical Foundation:
    DME preserves discriminative boundaries via pairwise differences D(Mi, Mj) = Mi - Mj.

    Medical Motivation:
    Key Clinical Differences (mode='key'):
    - T1CE - T1: Blood-brain barrier breakdown (active tumor enhancement)
    - FLAIR - T2: Perilesional edema specificity (infiltration)
    - T1CE - FLAIR: Enhancement vs edema differentiation

    All Pairwise Differences (mode='all'):
    - All C(4,2) = 6 combinations for complete differential encoding
    """

    def __init__(self, in_channels, out_channels=None):
        super().__init__()

        # ============================================
        # CONFIGURE THESE FOR ABLATION STUDIES
        # ============================================
        self.diff_mode = 'all'  # Change to 'key' or 'all'
        self.use_learnable_weights = True  # Change to False for fixed weights
        # ============================================

        if out_channels is None:
            out_channels = in_channels * 2

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Determine number of difference pairs based on mode
        if self.diff_mode == 'key':
            self.num_diffs = 3  # 3 clinically important pairs
            self.diff_names = ['T1CE-T1', 'FLAIR-T2', 'T1CE-FLAIR']
        elif self.diff_mode == 'all':
            self.num_diffs = 6  # All pairwise combinations C(4,2)
            self.diff_names = ['T1-T1CE', 'T1-T2', 'T1-FLAIR', 'T1CE-T2', 'T1CE-FLAIR', 'T2-FLAIR']

        # Learnable weights for different difference types
        if self.use_learnable_weights:
            self.diff_weights = nn.Parameter(torch.ones(self.num_diffs, 1, 1, 1, 1))

        # Differential encoder architecture
        encoder_in_channels = in_channels * self.num_diffs
        encoder_mid_channels = max(in_channels * 2, out_channels)

        self.diff_encoder = nn.Sequential(
            # First bottleneck
            nn.Conv3d(encoder_in_channels, encoder_mid_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(encoder_mid_channels),
            nn.GELU(),

            # Spatial processing with depthwise separable convolution
            nn.Conv3d(encoder_mid_channels, encoder_mid_channels, kernel_size=3, padding=1,
                      groups=encoder_mid_channels, bias=False),  # Depthwise
            nn.Conv3d(encoder_mid_channels, encoder_mid_channels, kernel_size=1, bias=False),  # Pointwise
            nn.InstanceNorm3d(encoder_mid_channels),
            nn.GELU(),

            # Second bottleneck
            nn.Conv3d(encoder_mid_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.GELU()
        )

    def compute_differences(self, t1, t1ce, t2, flair):
        """Compute pairwise differences based on selected mode"""
        if self.diff_mode == 'key':
            # 3 key clinical differences
            diffs = [
                t1ce - t1,  # Enhancement (BBB breakdown)
                flair - t2,  # Edema specificity (infiltration)
                t1ce - flair  # Enhancement vs edema differentiation
            ]
        elif self.diff_mode == 'all':
            # All 6 pairwise combinations
            diffs = [
                t1 - t1ce,  # Anatomical baseline vs enhancement
                t1 - t2,  # T1 vs T2 weighting
                t1 - flair,  # Normal vs fluid-sensitive
                t1ce - t2,  # Enhancement vs T2 intensity
                t1ce - flair,  # Enhancement vs edema
                t2 - flair  # T2 vs FLAIR specificity
            ]
        return diffs

    def forward(self, t1, t1ce, t2, flair):
        """Compute and encode pairwise differences"""
        # Compute differences based on mode
        diffs = self.compute_differences(t1, t1ce, t2, flair)

        # Apply learnable weights if enabled
        if self.use_learnable_weights:
            weighted_diffs = [d * w for d, w in zip(diffs, self.diff_weights)]
        else:
            weighted_diffs = diffs

        # Concatenate and encode
        concat_diffs = torch.cat(weighted_diffs, dim=1)
        encoded = self.diff_encoder(concat_diffs)

        return encoded

    def get_difference_maps(self, t1, t1ce, t2, flair):
        """Extract individual difference maps for visualization"""
        with torch.no_grad():
            diffs = self.compute_differences(t1, t1ce, t2, flair)

            if self.diff_mode == 'key':
                diff_maps = {
                    'T1CE-T1 (Enhancement)': diffs[0].cpu(),
                    'FLAIR-T2 (Edema Specificity)': diffs[1].cpu(),
                    'T1CE-FLAIR (Enhancement vs Edema)': diffs[2].cpu()
                }
            elif self.diff_mode == 'all':
                diff_maps = {
                    'T1-T1CE (Inverse Enhancement)': diffs[0].cpu(),
                    'T1-T2 (Weighting Difference)': diffs[1].cpu(),
                    'T1-FLAIR (Fluid Sensitivity)': diffs[2].cpu(),
                    'T1CE-T2 (Contrast Specificity)': diffs[3].cpu(),
                    'T1CE-FLAIR (Enhancement vs Edema)': diffs[4].cpu(),
                    'T2-FLAIR (Edema Specificity)': diffs[5].cpu()
                }
        return diff_maps

    def get_learned_weights(self):
        """Get learned importance weights for each difference pair"""
        if self.use_learnable_weights:
            weights = torch.softmax(self.diff_weights.squeeze(), dim=0)
            weight_dict = {name: weights[i].item()
                           for i, name in enumerate(self.diff_names)}
            return weight_dict
        return None


class DME1(nn.Module):
    """
    Differential Modality Encoding (DME)

    Theoretical Foundation:
    DME preserves discriminative boundaries via pairwise differences D(Mi, Mj) = Mi - Mj.

    Medical Motivation:
    Key Clinical Differences (mode='key'):
    - T1CE - T1: Blood-brain barrier breakdown (active tumor enhancement)
    - FLAIR - T2: Perilesional edema specificity (infiltration)
    - T1CE - FLAIR: Enhancement vs edema differentiation

    All Pairwise Differences (mode='all'):
    - All C(4,2) = 6 combinations for complete differential encoding
    """

    def __init__(self, in_channels, out_channels=None):
        super().__init__()

        # ============================================
        # CONFIGURE THESE FOR ABLATION STUDIES
        # ============================================
        self.diff_mode = 'key'  # Change to 'key' or 'all'
        self.use_learnable_weights = True  # Change to False for fixed weights
        # ============================================

        if out_channels is None:
            out_channels = in_channels * 2

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Determine number of difference pairs based on mode
        if self.diff_mode == 'key':
            self.num_diffs = 3  # 3 clinically important pairs
            self.diff_names = ['T1CE-T1', 'FLAIR-T2', 'T1CE-FLAIR']
        elif self.diff_mode == 'all':
            self.num_diffs = 6  # All pairwise combinations C(4,2)
            self.diff_names = ['T1-T1CE', 'T1-T2', 'T1-FLAIR', 'T1CE-T2', 'T1CE-FLAIR', 'T2-FLAIR']

        # Learnable weights for different difference types
        if self.use_learnable_weights:
            self.diff_weights = nn.Parameter(torch.ones(self.num_diffs, 1, 1, 1, 1))

        # Differential encoder architecture
        encoder_in_channels = in_channels * self.num_diffs
        encoder_mid_channels = max(in_channels * 2, out_channels)

        self.diff_encoder = nn.Sequential(
            # First bottleneck
            nn.Conv3d(encoder_in_channels, encoder_mid_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(encoder_mid_channels),
            nn.GELU(),

            # Spatial processing with depthwise separable convolution
            nn.Conv3d(encoder_mid_channels, encoder_mid_channels, kernel_size=3, padding=1,
                      groups=encoder_mid_channels, bias=False),  # Depthwise
            nn.Conv3d(encoder_mid_channels, encoder_mid_channels, kernel_size=1, bias=False),  # Pointwise
            nn.InstanceNorm3d(encoder_mid_channels),
            nn.GELU(),

            # Second bottleneck
            nn.Conv3d(encoder_mid_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.GELU()
        )

    def compute_differences(self, t1, t1ce, t2, flair):
        """Compute pairwise differences based on selected mode"""
        if self.diff_mode == 'key':
            # 3 key clinical differences
            diffs = [
                t1ce - t1,  # Enhancement (BBB breakdown)
                flair - t2,  # Edema specificity (infiltration)
                t1ce - flair  # Enhancement vs edema differentiation
            ]
        elif self.diff_mode == 'all':
            # All 6 pairwise combinations
            diffs = [
                t1 - t1ce,  # Anatomical baseline vs enhancement
                t1 - t2,  # T1 vs T2 weighting
                t1 - flair,  # Normal vs fluid-sensitive
                t1ce - t2,  # Enhancement vs T2 intensity
                t1ce - flair,  # Enhancement vs edema
                t2 - flair  # T2 vs FLAIR specificity
            ]
        return diffs

    def forward(self, t1, t1ce, t2, flair):
        """Compute and encode pairwise differences"""
        # Compute differences based on mode
        diffs = self.compute_differences(t1, t1ce, t2, flair)

        # Apply learnable weights if enabled
        if self.use_learnable_weights:
            weighted_diffs = [d * w for d, w in zip(diffs, self.diff_weights)]
        else:
            weighted_diffs = diffs

        # Concatenate and encode
        concat_diffs = torch.cat(weighted_diffs, dim=1)
        encoded = self.diff_encoder(concat_diffs)

        return encoded

    def get_difference_maps(self, t1, t1ce, t2, flair):
        """Extract individual difference maps for visualization"""
        with torch.no_grad():
            diffs = self.compute_differences(t1, t1ce, t2, flair)

            if self.diff_mode == 'key':
                diff_maps = {
                    'T1CE-T1 (Enhancement)': diffs[0].cpu(),
                    'FLAIR-T2 (Edema Specificity)': diffs[1].cpu(),
                    'T1CE-FLAIR (Enhancement vs Edema)': diffs[2].cpu()
                }
            elif self.diff_mode == 'all':
                diff_maps = {
                    'T1-T1CE (Inverse Enhancement)': diffs[0].cpu(),
                    'T1-T2 (Weighting Difference)': diffs[1].cpu(),
                    'T1-FLAIR (Fluid Sensitivity)': diffs[2].cpu(),
                    'T1CE-T2 (Contrast Specificity)': diffs[3].cpu(),
                    'T1CE-FLAIR (Enhancement vs Edema)': diffs[4].cpu(),
                    'T2-FLAIR (Edema Specificity)': diffs[5].cpu()
                }
        return diff_maps

    def get_learned_weights(self):
        """Get learned importance weights for each difference pair"""
        if self.use_learnable_weights:
            weights = torch.softmax(self.diff_weights.squeeze(), dim=0)
            weight_dict = {name: weights[i].item()
                           for i, name in enumerate(self.diff_names)}
            return weight_dict
        return None

class DME2(nn.Module):
    """
    Differential Modality Encoding (DME) for Skip Connections

    Theoretical Foundation:
    DME preserves discriminative boundaries via pairwise differences D(Mi, Mj) = Mi - Mj.

    Medical Motivation:
    Key Clinical Differences (mode='key'):
    - T1CE - T1: Blood-brain barrier breakdown (active tumor enhancement)
    - FLAIR - T2: Perilesional edema specificity (infiltration)
    - T1CE - FLAIR: Enhancement vs edema differentiation

    All Pairwise Differences (mode='all'):
    - All C(4,2) = 6 combinations for complete differential encoding
    """

    def __init__(self, in_channels, out_channels=None):
        super().__init__()

        # ============================================
        # CONFIGURE THESE FOR ABLATION STUDIES
        # ============================================
        self.diff_mode = 'all'  # Change to 'key' or 'all'
        self.use_learnable_weights = False  # Change to False for fixed weights
        # ============================================

        if out_channels is None:
            out_channels = in_channels * 2

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Determine number of difference pairs based on mode
        if self.diff_mode == 'key':
            self.num_diffs = 3  # 3 clinically important pairs
            self.diff_names = ['T1CE-T1', 'FLAIR-T2', 'T1CE-FLAIR']
        elif self.diff_mode == 'all':
            self.num_diffs = 6  # All pairwise combinations C(4,2)
            self.diff_names = ['T1-T1CE', 'T1-T2', 'T1-FLAIR', 'T1CE-T2', 'T1CE-FLAIR', 'T2-FLAIR']

        # Learnable weights for different difference types
        if self.use_learnable_weights:
            self.diff_weights = nn.Parameter(torch.ones(self.num_diffs, 1, 1, 1, 1))

        # Differential encoder architecture
        encoder_in_channels = in_channels * self.num_diffs
        encoder_mid_channels = max(in_channels * 2, out_channels)

        self.diff_encoder = nn.Sequential(
            # First bottleneck
            nn.Conv3d(encoder_in_channels, encoder_mid_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(encoder_mid_channels),
            nn.GELU(),

            # Spatial processing with depthwise separable convolution
            nn.Conv3d(encoder_mid_channels, encoder_mid_channels, kernel_size=3, padding=1,
                      groups=encoder_mid_channels, bias=False),  # Depthwise
            nn.Conv3d(encoder_mid_channels, encoder_mid_channels, kernel_size=1, bias=False),  # Pointwise
            nn.InstanceNorm3d(encoder_mid_channels),
            nn.GELU(),

            # Second bottleneck
            nn.Conv3d(encoder_mid_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.GELU()
        )

    def compute_differences(self, t1, t1ce, t2, flair):
        """Compute pairwise differences based on selected mode"""
        if self.diff_mode == 'key':
            # 3 key clinical differences
            diffs = [
                t1ce - t1,  # Enhancement (BBB breakdown)
                flair - t2,  # Edema specificity (infiltration)
                t1ce - flair  # Enhancement vs edema differentiation
            ]
        elif self.diff_mode == 'all':
            # All 6 pairwise combinations
            diffs = [
                t1 - t1ce,  # Anatomical baseline vs enhancement
                t1 - t2,  # T1 vs T2 weighting
                t1 - flair,  # Normal vs fluid-sensitive
                t1ce - t2,  # Enhancement vs T2 intensity
                t1ce - flair,  # Enhancement vs edema
                t2 - flair  # T2 vs FLAIR specificity
            ]
        return diffs

    def forward(self, t1, t1ce, t2, flair):
        """Compute and encode pairwise differences"""
        # Compute differences based on mode
        diffs = self.compute_differences(t1, t1ce, t2, flair)

        # Apply learnable weights if enabled
        if self.use_learnable_weights:
            weighted_diffs = [d * w for d, w in zip(diffs, self.diff_weights)]
        else:
            weighted_diffs = diffs

        # Concatenate and encode
        concat_diffs = torch.cat(weighted_diffs, dim=1)
        encoded = self.diff_encoder(concat_diffs)

        return encoded

    def get_difference_maps(self, t1, t1ce, t2, flair):
        """Extract individual difference maps for visualization"""
        with torch.no_grad():
            diffs = self.compute_differences(t1, t1ce, t2, flair)

            if self.diff_mode == 'key':
                diff_maps = {
                    'T1CE-T1 (Enhancement)': diffs[0].cpu(),
                    'FLAIR-T2 (Edema Specificity)': diffs[1].cpu(),
                    'T1CE-FLAIR (Enhancement vs Edema)': diffs[2].cpu()
                }
            elif self.diff_mode == 'all':
                diff_maps = {
                    'T1-T1CE (Inverse Enhancement)': diffs[0].cpu(),
                    'T1-T2 (Weighting Difference)': diffs[1].cpu(),
                    'T1-FLAIR (Fluid Sensitivity)': diffs[2].cpu(),
                    'T1CE-T2 (Contrast Specificity)': diffs[3].cpu(),
                    'T1CE-FLAIR (Enhancement vs Edema)': diffs[4].cpu(),
                    'T2-FLAIR (Edema Specificity)': diffs[5].cpu()
                }
        return diff_maps

    def get_learned_weights(self):
        """Get learned importance weights for each difference pair"""
        if self.use_learnable_weights:
            weights = torch.softmax(self.diff_weights.squeeze(), dim=0)
            weight_dict = {name: weights[i].item()
                           for i, name in enumerate(self.diff_names)}
            return weight_dict
        return None

class DME3(nn.Module):

    def __init__(self, in_channels, out_channels=None):
        super().__init__()

        # ============================================
        # CONFIGURE THESE FOR ABLATION STUDIES
        # ============================================
        self.diff_mode = 'key'  # Change to 'key' or 'all'
        self.use_learnable_weights = False  # Change to False for fixed weights
        # ============================================

        if out_channels is None:
            out_channels = in_channels * 2

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Determine number of difference pairs based on mode
        if self.diff_mode == 'key':
            self.num_diffs = 3  # 3 clinically important pairs
            self.diff_names = ['T1CE-T1', 'FLAIR-T2', 'T1CE-FLAIR']
        elif self.diff_mode == 'all':
            self.num_diffs = 6  # All pairwise combinations C(4,2)
            self.diff_names = ['T1-T1CE', 'T1-T2', 'T1-FLAIR', 'T1CE-T2', 'T1CE-FLAIR', 'T2-FLAIR']

        # Learnable weights for different difference types
        if self.use_learnable_weights:
            self.diff_weights = nn.Parameter(torch.ones(self.num_diffs, 1, 1, 1, 1))

        # Differential encoder architecture
        encoder_in_channels = in_channels * self.num_diffs
        encoder_mid_channels = max(in_channels * 2, out_channels)

        self.diff_encoder = nn.Sequential(
            # First bottleneck
            nn.Conv3d(encoder_in_channels, encoder_mid_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(encoder_mid_channels),
            nn.GELU(),

            # Spatial processing with depthwise separable convolution
            nn.Conv3d(encoder_mid_channels, encoder_mid_channels, kernel_size=3, padding=1,
                      groups=encoder_mid_channels, bias=False),  # Depthwise
            nn.Conv3d(encoder_mid_channels, encoder_mid_channels, kernel_size=1, bias=False),  # Pointwise
            nn.InstanceNorm3d(encoder_mid_channels),
            nn.GELU(),

            # Second bottleneck
            nn.Conv3d(encoder_mid_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.GELU()
        )

    def compute_differences(self, t1, t1ce, t2, flair):
        """Compute pairwise differences based on selected mode"""
        if self.diff_mode == 'key':
            # 3 key clinical differences
            diffs = [
                t1ce - t1,  # Enhancement (BBB breakdown)
                flair - t2,  # Edema specificity (infiltration)
                t1ce - flair  # Enhancement vs edema differentiation
            ]
        elif self.diff_mode == 'all':
            # All 6 pairwise combinations
            diffs = [
                t1 - t1ce,  # Anatomical baseline vs enhancement
                t1 - t2,  # T1 vs T2 weighting
                t1 - flair,  # Normal vs fluid-sensitive
                t1ce - t2,  # Enhancement vs T2 intensity
                t1ce - flair,  # Enhancement vs edema
                t2 - flair  # T2 vs FLAIR specificity
            ]
        return diffs

    def forward(self, t1, t1ce, t2, flair):
        """Compute and encode pairwise differences"""
        # Compute differences based on mode
        diffs = self.compute_differences(t1, t1ce, t2, flair)

        # Apply learnable weights if enabled
        if self.use_learnable_weights:
            weighted_diffs = [d * w for d, w in zip(diffs, self.diff_weights)]
        else:
            weighted_diffs = diffs

        # Concatenate and encode
        concat_diffs = torch.cat(weighted_diffs, dim=1)
        encoded = self.diff_encoder(concat_diffs)

        return encoded

    def get_difference_maps(self, t1, t1ce, t2, flair):
        """Extract individual difference maps for visualization"""
        with torch.no_grad():
            diffs = self.compute_differences(t1, t1ce, t2, flair)

            if self.diff_mode == 'key':
                diff_maps = {
                    'T1CE-T1 (Enhancement)': diffs[0].cpu(),
                    'FLAIR-T2 (Edema Specificity)': diffs[1].cpu(),
                    'T1CE-FLAIR (Enhancement vs Edema)': diffs[2].cpu()
                }
            elif self.diff_mode == 'all':
                diff_maps = {
                    'T1-T1CE (Inverse Enhancement)': diffs[0].cpu(),
                    'T1-T2 (Weighting Difference)': diffs[1].cpu(),
                    'T1-FLAIR (Fluid Sensitivity)': diffs[2].cpu(),
                    'T1CE-T2 (Contrast Specificity)': diffs[3].cpu(),
                    'T1CE-FLAIR (Enhancement vs Edema)': diffs[4].cpu(),
                    'T2-FLAIR (Edema Specificity)': diffs[5].cpu()
                }
        return diff_maps

    def get_learned_weights(self):
        """Get learned importance weights for each difference pair"""
        if self.use_learnable_weights:
            weights = torch.softmax(self.diff_weights.squeeze(), dim=0)
            weight_dict = {name: weights[i].item()
                           for i, name in enumerate(self.diff_names)}
            return weight_dict
        return None
