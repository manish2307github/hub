"""
AGHNN - Adaptive Green Hybrid Neural Network
Main model implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .components import (
    EfficientStem,
    ComplexityEstimator,
    EasyPath,
    HardPath,
    AdaptiveFusion
)


class AGHNN(nn.Module):
    """
    Adaptive Green Hybrid Neural Network
    
    A novel architecture for energy-efficient deep learning that:
    1. Adaptively routes inputs through easy/hard paths based on complexity
    2. Uses green regularization for energy-aware training
    3. Combines multiple efficiency techniques in a hybrid design
    
    Args:
        num_classes: Number of output classes
        stem_channels: Channels in stem block
        easy_channels: Base channels for easy path
        hard_channels: Base channels for hard path
        easy_blocks: Number of blocks per stage in easy path
        hard_blocks: Number of blocks per stage in hard path
        threshold_easy: Complexity threshold for easy path
        threshold_hard: Complexity threshold for hard path
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        stem_channels: int = 32,
        easy_channels: int = 32,
        hard_channels: int = 64,
        easy_blocks: list = [1, 1, 1],
        hard_blocks: list = [2, 3, 3],
        threshold_easy: float = 0.3,
        threshold_hard: float = 0.7,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.threshold_easy = threshold_easy
        self.threshold_hard = threshold_hard
        
        # Stem for initial feature extraction
        self.stem = EfficientStem(in_channels=3, out_channels=stem_channels)
        
        # Complexity estimator
        self.complexity_estimator = ComplexityEstimator(
            in_channels=stem_channels,
            hidden_channels=32
        )
        
        # Adaptive paths
        self.easy_path = EasyPath(
            in_channels=stem_channels,
            num_classes=num_classes,
            base_channels=easy_channels,
            num_blocks=easy_blocks
        )
        
        self.hard_path = HardPath(
            in_channels=stem_channels,
            num_classes=num_classes,
            base_channels=hard_channels,
            num_blocks=hard_blocks
        )
        
        # ✅ Adaptive fusion module (paths have their own FC layers)
        self.fusion = AdaptiveFusion(num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        return_complexity: bool = False,
        hard_routing: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with adaptive routing
        
        Args:
            x: Input tensor [B, 3, H, W]
            return_complexity: If True, return complexity scores
            hard_routing: If True, use hard routing (inference mode)
            
        Returns:
            Dictionary containing:
                - logits: Final predictions
                - easy_logits: Easy path predictions (for distillation)
                - hard_logits: Hard path predictions (for distillation)
                - complexity: Complexity scores (if return_complexity=True)
        """
        # Extract stem features
        stem_features = self.stem(x)
        
        # Estimate input complexity
        complexity = self.complexity_estimator(stem_features)
        
        # Forward through both paths
        easy_logits, easy_features = self.easy_path(stem_features)
        hard_logits, hard_features = self.hard_path(stem_features)
        
        # Apply dropout to features
        easy_logits = self.dropout(easy_logits)
        hard_logits = self.dropout(hard_logits)
        
        # Fuse outputs based on complexity
        logits = self.fusion(
            easy_logits,
            hard_logits,
            complexity,
            hard_routing=hard_routing,
            threshold=self.threshold_hard
        )
        
        outputs = {
            'logits': logits,
            'easy_logits': easy_logits,
            'hard_logits': hard_logits,
        }
        
        if return_complexity:
            outputs['complexity'] = complexity
            
        return outputs
    
    def get_flops(self, input_size: Tuple[int, int, int, int] = (1, 3, 32, 32)) -> Dict[str, int]:
        """
        Estimate FLOPs for different paths
        
        Args:
            input_size: Input tensor size (B, C, H, W)
            
        Returns:
            Dictionary with FLOPs for each component
        """
        from thop import profile
        
        dummy_input = torch.randn(input_size)
        
        # This is a simplified estimation
        # Full implementation would trace each path separately
        total_flops, _ = profile(self, inputs=(dummy_input,), verbose=False)
        
        return {
            'total': int(total_flops),
            'easy_path': int(total_flops * 0.4),  # Approximate
            'hard_path': int(total_flops * 0.6),  # Approximate
        }
    
    def compute_energy_loss(
        self,
        complexity: torch.Tensor,
        easy_energy: float = 1.0,
        hard_energy: float = 2.5
    ) -> torch.Tensor:
        """
        Compute energy loss based on path selection
        
        Args:
            complexity: Complexity scores [B] (squeezed)
            easy_energy: Relative energy cost of easy path
            hard_energy: Relative energy cost of hard path
            
        Returns:
            Energy loss tensor
        """
        # Ensure complexity is [B] shaped
        if complexity.dim() > 1:
            complexity = complexity.squeeze(-1)
        # Expected energy consumption
        expected_energy = complexity * hard_energy + (1 - complexity) * easy_energy
        return expected_energy.mean()


# Model variants

def AGHNN_Tiny(num_classes: int = 10) -> AGHNN:
    """
    AGHNN-Tiny: ~0.8M parameters, ~150M FLOPs
    Best for edge devices with strict constraints
    """
    return AGHNN(
        num_classes=num_classes,
        stem_channels=16,
        easy_channels=16,
        hard_channels=32,
        easy_blocks=[1, 1, 1],
        hard_blocks=[1, 2, 2],
    )

def AGHNN_Small(num_classes: int = 10) -> AGHNN:
    """
    AGHNN-Small: ~3.5M parameters, ~600M FLOPs
    Good balance for mobile applications (FIXED: increased easy path capacity)
    """
    return AGHNN(
        num_classes=num_classes,
        stem_channels=32,
        easy_channels=48,  # Increased from 32 to 48
        hard_channels=64,
        easy_blocks=[1, 2, 2],  # Increased from [1,1,1] to [1,2,2]
        hard_blocks=[2, 2, 2],
    )


def AGHNN_Base(num_classes: int = 10) -> AGHNN:
    """
    AGHNN-Base: ~5.2M parameters, ~900M FLOPs
    Standard configuration for most tasks
    """
    return AGHNN(
        num_classes=num_classes,
        stem_channels=32,
        easy_channels=48,
        hard_channels=96,
        easy_blocks=[2, 2, 2],
        hard_blocks=[3, 4, 4],
    )


def AGHNN_Large(num_classes: int = 10) -> AGHNN:
    """
    AGHNN-Large: ~12M parameters, ~2.1G FLOPs
    Maximum capacity for challenging tasks
    """
    return AGHNN(
        num_classes=num_classes,
        stem_channels=48,
        easy_channels=64,
        hard_channels=128,
        easy_blocks=[2, 3, 3],
        hard_blocks=[4, 6, 6],
    )
