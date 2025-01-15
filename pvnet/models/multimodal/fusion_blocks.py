# fusion_blocks.py

""" 
Fusion blocks for dynamic multimodal fusion implementation

Definition of foundational fusion mechanisms; DynamicFusionModule and ModalityGating

Aformentioned fusion blocks apply dynamic attention, weighted combinations and / or gating mechanisms for feature learning

Summararily, this enables dynamic feature learning through attention based weighting and modality specific gating
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

    """ Implementation of dynamic multimodal fusion through cross attention and weighted combination """

    # Input dimension specified and common embedding dimension
    # Quantity of attention heads also specified
    def __init__(
        self,
        feature_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        fusion_method: str = "weighted_sum",
        use_residual: bool = True
    ):
        super().__init__()
        
        if hidden_dim <= 0 or num_heads <= 0:
            raise ValueError("hidden_dim and num_heads must be positive")
            
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        self.fusion_method = fusion_method
        self.use_residual = use_residual
        
        if fusion_method not in ["weighted_sum", "concat"]:
            raise ValueError(f"Invalid fusion method: {fusion_method}")
        
        # Projections - modality specific
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
        
        # Attention - cross modal
        self.cross_attention = MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Weight computation network definition
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
        """ Validates input feature dimensions and sequence lengths """

        if not features:
            raise ValueError("Empty features dict")
            
        seq_length = None
        for name, feat in features.items():
            if feat is None:
                raise ValueError(f"None tensor for modality: {name}")
                
            if seq_length is None:
                seq_length = feat.size(1)
            elif feat.size(1) != seq_length:
                raise ValueError("All modalities must have same sequence length")

    def compute_modality_weights(
        self,
        features: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        """ Computation of attention weights for each feature """

        batch_size, seq_len = features.size(0), features.size(1)
        flat_features = features.reshape(-1, features.size(-1))
        weights = self.weight_network(flat_features)
        weights = weights.reshape(batch_size, seq_len, 1)
        
        if modality_mask is not None:
            weights = weights.reshape(batch_size, -1, 1)[:, :modality_mask.size(1), :]
            weights = weights.masked_fill(~modality_mask.unsqueeze(-1), 0.0)
            
        # Normalisation of weights
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)
        return weights

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        """ Forward pass for dynamic fusion """

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
            
        # Stack features
        feature_stack = torch.stack(projected_features, dim=1)
        
        # Apply cross attention
        attended_features = []
        for i in range(feature_stack.size(1)):
            query = feature_stack[:, i]
            key_value = feature_stack[:, [j for j in range(feature_stack.size(1)) if j != i]]
            if key_value.size(1) > 0:
                attended = self.cross_attention(query, key_value.reshape(-1, seq_len, self.hidden_dim), 
                                            key_value.reshape(-1, seq_len, self.hidden_dim))
                attended_features.append(attended)
            else:
                attended_features.append(query)
                
        # Average across modalities
        attended_features = torch.stack(attended_features, dim=1)
        attended_avg = attended_features.mean(dim=1)
        
        # Mask attended features to match
        if modality_mask is not None:
            # Create binary mask matching sequence length
            seq_mask = torch.zeros((batch_size, seq_len), device=attended_avg.device).bool()
            seq_mask[:, :modality_mask.size(1)] = modality_mask
            
            # Compute weights on masked features
            weights = self.compute_modality_weights(attended_avg, seq_mask)
            weights = weights.unsqueeze(1).expand(-1, attended_features.size(1), -1, 1)
        else:
            weights = self.compute_modality_weights(attended_avg)
            weights = weights.unsqueeze(1).expand(-1, attended_features.size(1), -1, 1)
        
        # Application of weighted features
        weighted_features = attended_features * weights
        
        if self.fusion_method == "weighted_sum":
            fused = weighted_features.sum(dim=1)
        else:
            concat = weighted_features.reshape(batch_size, seq_len, -1)
            fused = self.output_projection(concat)
            
        # Application of residual
        if self.use_residual:
            residual = feature_stack.mean(dim=1)
            fused = self.layer_norm(fused + residual)
            
        # Collapse sequence dimension for output
        fused = fused.mean(dim=1)
            
        return fused




class ModalityGating(AbstractFusionBlock):
    """ Implementation of modality specific gating mechanism """

    # Input and hidden dimension definition
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
        
        # Define gate networks for each modality
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
        """ Validation helper for input feature dict """

        if not features:
            raise ValueError("Empty features dict")
        for name, feat in features.items():
            if feat is None:
                raise ValueError(f"None tensor for modality: {name}")


    def forward(
        self,
        features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        """ Application of modality specific gating """

        self._validate_features(features)
        gated_features = {}
        
        for name, feat in features.items():
            if feat is not None and name in self.gate_networks:
                batch_size, seq_len, feat_dim = feat.shape

                # Gate computation sequence 
                flat_feat = feat.reshape(-1, feat_dim)                
                gate = self.gate_networks[name](flat_feat)                
                gate = gate.reshape(batch_size, seq_len, 1)
                
                # Application of gating
                gated_features[name] = feat * gate
                
        return gated_features
