# fusion_blocks.py

""" 
Fusion blocks for dynamic multimodal fusion implementation

Definition of foundational fusion mechanisms; DynamicFusionModule and ModalityGating

Aformentioned fusion blocks apply dynamic attention, weighted combinations and / or gating mechanisms for feature learning
"""


import torch
import torch.nn.functional as F
from torch import nn
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, List

from pvnet.models.multimodal.attention_blocks import MultiheadAttention


class AbstractFusionBlock(nn.Module, ABC):
    """ Abstract fusion base class definition """

    # Forward pass
    @abstractmethod
    def forward(
        self,
        features: Dict[str, torch.Tensor]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        pass


class DynamicFusionModule(AbstractFusionBlock):

    """ Dynamic fusion implementation / definition """
    def __init__(
        self,
        feature_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        fusion_method: str = "weighted_sum",
        use_residual: bool = True
    ):
        
        # Initialisation of dynamic fusion module
        super().__init__()
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        self.fusion_method = fusion_method
        self.use_residual = use_residual
        
        if fusion_method not in ["weighted_sum", "concat"]:
            raise ValueError(f"Invalid fusion method: {fusion_method}")
        
        # Define projections for each modality
        # Specified features only considered
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for name, dim in feature_dims.items()
            if dim > 0
        })
        
        # Cross attention mechanism
        self.cross_attention = MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Dynamic weighting network
        # Weight generation per modality - consistently positive
        self.weight_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Optional output projection for concatenation
        if fusion_method == "concat":
            self.output_projection = nn.Sequential(
                nn.Linear(hidden_dim * len(feature_dims), hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # Layer normalisation for residual connections
        if use_residual:
            self.layer_norm = nn.LayerNorm(hidden_dim)
            

    def compute_modality_weights(
        self,
        attended_features: torch.Tensor,
        available_modalities: List[str],
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # Computation of dynamic weights for available modalities
        batch_size = attended_features.size(0)
        num_modalities = len(available_modalities)
        
        # Independent weight generation per modality
        weights = self.weight_network(attended_features)
        
        if mask is not None:
            # Reshape mask to match weights dimension
            mask = mask.unsqueeze(-1)
            weights = weights.masked_fill(~mask, 0.0)
        
        # Normalise weights
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)
        
        return weights


    def forward(
        self,
        features: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # Forward pass for dynamic fusion
        # Project each modality to common space
        projected_features = {
            name: self.projections[name](feat)
            for name, feat in features.items()
            if feat is not None and self.feature_dims[name] > 0
        }
        
        if not projected_features:
            raise ValueError("Invalid features")
        
        # Stack features for attention and store for residual connection
        feature_stack = torch.stack(list(projected_features.values()), dim=1)        
        input_features = feature_stack
        
        # Cross attention - application
        attended_features = self.cross_attention(
            feature_stack,
            feature_stack,
            feature_stack
        )
        
        # Apply dynamic weights
        weights = self.compute_modality_weights(
            attended_features, 
            list(projected_features.keys()),
            mask
        )
        
        # Weighted sum or concatenation
        if self.fusion_method == "weighted_sum":
            weighted_features = attended_features * weights
            fused_features = weighted_features.sum(dim=1)
        else:
            weighted_features = attended_features * weights
            fused_features = self.output_projection(
                weighted_features.view(weighted_features.size(0), -1)
            )
        
        # Apply residual connection
        if self.use_residual:
            residual = input_features.mean(dim=1)
            fused_features = self.layer_norm(fused_features + residual)
            
        return fused_features


class ModalityGating(AbstractFusionBlock):

    """ Modality gating mechanism definition """
    def __init__(
        self,
        feature_dims: Dict[str, int],
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        # Initialisation of modality gating module
        super().__init__()
        self.feature_dims = feature_dims
        
        # Create gate networks for each modality
        self.gate_networks = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            for name, dim in feature_dims.items()
            if dim > 0
        })
        
    def forward(
        self,
        features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        # Forward pass for modality gating
        gated_features = {}
        
        #  Gate value and subsequent application
        for name, feat in features.items():
            if feat is not None and self.feature_dims.get(name, 0) > 0:
                gate = self.gate_networks[name](feat)
                gated_features[name] = feat * gate
                
        return gated_features
