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
    def __init__(
        self,
        feature_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        fusion_method: str = "weighted_sum",
        use_residual: bool = True
    ):
        nn.Module.__init__(self)
        
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
            
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        self.fusion_method = fusion_method
        self.use_residual = use_residual
        
        if fusion_method not in ["weighted_sum", "concat"]:
            raise ValueError(f"Invalid fusion method: {fusion_method}")
        
        # Projections
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
        
        # Attention
        self.cross_attention = MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Weight network
        self.weight_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Optional concat projection
        if fusion_method == "concat":
            self.output_projection = nn.Sequential(
                nn.Linear(hidden_dim * len(feature_dims), hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        if use_residual:
            self.layer_norm = nn.LayerNorm(hidden_dim)
            
    def _validate_features(self, features: Dict[str, torch.Tensor]) -> None:
        if not features:
            raise ValueError("Empty features dictionary")
            
        seq_length = None
        for name, feat in features.items():
            if feat is None:
                raise ValueError(f"None tensor for modality: {name}")
                
            if seq_length is None:
                seq_length = feat.size(1)
            elif feat.size(1) != seq_length:
                raise ValueError("All modalities must have the same sequence length")

    def compute_modality_weights(
        self,
        features: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute weights for each feature.
        
        Args:
            features: [batch_size, seq_len, hidden_dim] tensor
            modality_mask: Optional attention mask
            
        Returns:
            [batch_size, seq_len, 1] tensor of weights
        """
        # Compute weights for each feature
        flat_features = features.reshape(-1, features.size(-1))  # [B*S, H]
        weights = self.weight_network(flat_features)  # [B*S, 1]
        weights = weights.reshape(features.size(0), features.size(1), 1)  # [B, S, 1]
        
        if modality_mask is not None:
            modality_mask = modality_mask.unsqueeze(-1)  # [B, S, 1]
            weights = weights.masked_fill(~modality_mask, 0.0)
            
        # Normalize weights
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)
        return weights

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass
        
        Args:
            features: Dict of [batch_size, seq_len, feature_dim] tensors
            modality_mask: Optional attention mask
            
        Returns:
            [batch_size, hidden_dim] tensor if seq_len=1, else [batch_size, seq_len, hidden_dim]
        """
        self._validate_features(features)
        
        batch_size = next(iter(features.values())).size(0)
        seq_len = next(iter(features.values())).size(1)
        
        # Project each modality
        projected_features = []
        for name, feat in features.items():
            if self.feature_dims[name] > 0:
                flat_feat = feat.reshape(-1, feat.size(-1))
                proj = self.projections[name](flat_feat)
                projected = proj.reshape(batch_size, seq_len, self.hidden_dim)
                projected_features.append(projected)
                
        if not projected_features:
            raise ValueError("No valid features after projection")
            
        # Stack and apply attention
        feature_stack = torch.stack(projected_features, dim=2)  # [B, S, M, H]
        
        # Cross attention
        attended_features = self.cross_attention(
            feature_stack, feature_stack, feature_stack
        )  # [B, S, M, H]
        
        # Average across modalities first
        attended_avg = attended_features.mean(dim=2)  # [B, S, H]
        
        # Compute weights on averaged features
        weights = self.compute_modality_weights(attended_avg, modality_mask)  # [B, S, 1]
        
        # Apply weights
        weighted_features = attended_features * weights.unsqueeze(2)  # [B, S, M, H]
        
        if self.fusion_method == "weighted_sum":
            # Sum across modalities
            fused = weighted_features.sum(dim=2)  # [B, S, H]
        else:
            # Concatenate modalities
            concat = weighted_features.reshape(batch_size, seq_len, -1)  # [B, S, M*H]
            fused = self.output_projection(concat)  # [B, S, H]
            
        # Apply residual if needed    
        if self.use_residual:
            residual = feature_stack.mean(dim=2)  # [B, S, H]
            fused = self.layer_norm(fused + residual)
            
        # Remove sequence dimension if length is 1    
        if seq_len == 1:
            fused = fused.squeeze(1)
            
        return fused


class ModalityGating(AbstractFusionBlock):
    def __init__(
        self,
        feature_dims: Dict[str, int],
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        
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

    def _validate_features(self, features: Dict[str, torch.Tensor]) -> None:

        if not features:
            raise ValueError("Empty features dictionary")
        for name, feat in features.items():
            if feat is None:
                raise ValueError(f"None tensor for modality: {name}")


    def forward(
        self,
        features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        self._validate_features(features)

        gated_features = {}
        
        for name, feat in features.items():
            if feat is not None and name in self.gate_networks:
                # Handle 3D tensors (batch_size, sequence_length, feature_dim)
                batch_size, seq_len, feat_dim = feat.shape
                
                # Reshape to (batch_size * seq_len, feature_dim)
                flat_feat = feat.reshape(-1, feat_dim)
                
                # Compute gates
                gate = self.gate_networks[name](flat_feat)
                
                # Reshape gates back to match input
                gate = gate.reshape(batch_size, seq_len, 1)
                
                # Apply gating
                gated_features[name] = feat * gate
                
        return gated_features
