# fusion_blocks.py

""" 
Fusion blocks for dynamic multimodal fusion implementation

Definition of foundational fusion mechanisms; DynamicFusionModule and ModalityGating

Aformentioned fusion blocks apply dynamic attention, weighted combinations and / or gating mechanisms for feature learning

Summararily - enables dynamic feature learning through attention based weighting and modality specific gating
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
    # Function mapping 
    # F: X → Y in fusion space
    @abstractmethod
    def forward(
        self,
        features: Dict[str, torch.Tensor]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        pass


class DynamicFusionModule(AbstractFusionBlock):

    """ Implementation of dynamic multimodal fusion through cross attention and weighted combination """

    # Define feature dimensions 
    # d_i ∈ ℝ^n
    # Shared latent space 
    # ℝ^h
    # Attention mechanisms 
    # A_i: ℝ^d → ℝ^{d/h}
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
        # φ_m: ℝ^{d_m} → ℝ^h for m ∈ M
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
        # ℝ^{d_m} → ℝ^h
        self.cross_attention = MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Weight computation network definition - dynamic
        # W: ℝ^h → [0,1]
        self.weight_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Optional concat projection
        # P: ℝ^{h|M|} → ℝ^h
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
        
        # Validate feature space dimensionality d_m
        # Validate sequence length L
        if not isinstance(features, dict) or not features:
            if isinstance(features, torch.Tensor):
                return  # Skip validation for single tensor
            raise ValueError("Empty features dict")
        
        # Validate temporal dimensions L_m across modalities
        multi_dim_features = {}
        for name, feat in features.items():
            if feat is None:
                raise ValueError(f"None tensor for modality: {name}")
            
            if feat.ndim > 1:
                multi_dim_features[name] = feat.size(1)
        
        # Verification step 
        # L_i = L_j ∀i,j ∈ M
        feature_lengths = set(multi_dim_features.values())
        if len(feature_lengths) > 1:
            raise ValueError(f"All modalities must have same sequence length. Current lengths: {multi_dim_features}")

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
        # α_m = w_m / Σ_j w_j
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
        
        # Apply modality-specific embeddings 
        # φ_m(x_m)
        projected_features = []
        for name, feat in features.items():
            if self.feature_dims[name] > 0:
                flat_feat = feat.reshape(-1, feat.size(-1))
                proj = self.projections[name](flat_feat)
                projected = proj.reshape(batch_size, seq_len, self.hidden_dim)
                projected_features.append(projected)
                
        if not projected_features:
            raise ValueError("No valid features after projection")
            
        # Tensor product of embedded features 
        # ⊗_{m∈M} φ_m(x_m)
        feature_stack = torch.stack(projected_features, dim=1)

        # Apply cross attention
        # A(⊗_{m∈M} φ_m(x_m))
        attended_features = []
        for i in range(feature_stack.size(1)):
            query = feature_stack[:, i]
            if feature_stack.size(1) > 1:

                # Case |M| > 1
                # Apply cross-modal attention A_c
                key_value = feature_stack[:, [j for j in range(feature_stack.size(1)) if j != i]]
                attended = self.cross_attention(query, key_value.reshape(-1, seq_len, self.hidden_dim), 
                                            key_value.reshape(-1, seq_len, self.hidden_dim))
            else:

                # Case |M| = 1
                # Apply self-attention A_s
                attended = self.cross_attention(query, query, query)
            attended_features.append(attended)
    
        # Compute mean representation
        # μ = 1/|M| Σ_{m∈M} A_m
        attended_features = torch.stack(attended_features, dim=1)
        attended_avg = attended_features.mean(dim=1)
        
        # Apply attention mask 
        # M ∈ {0,1}^{B×L}
        if modality_mask is not None:
            seq_mask = torch.zeros((batch_size, seq_len), device=attended_avg.device).bool()
            seq_mask[:, :modality_mask.size(1)] = modality_mask            
            weights = self.compute_modality_weights(attended_avg, seq_mask)
            weights = weights.unsqueeze(1).expand(-1, attended_features.size(1), -1, 1)
        else:
            weights = self.compute_modality_weights(attended_avg)
            weights = weights.unsqueeze(1).expand(-1, attended_features.size(1), -1, 1)
        
        # Apply dynamic modality weights 
        # w_m ∈ [0,1]
        weighted_features = attended_features * weights
        
        if self.fusion_method == "weighted_sum":
            fused = weighted_features.sum(dim=1)
        else:
            concat = weighted_features.reshape(batch_size, seq_len, -1)
            fused = self.output_projection(concat)
            
        # Application of residual
        # r(x) = LayerNorm(x + μ(x))
        if self.use_residual:
            residual = feature_stack.mean(dim=1)
            fused = self.layer_norm(fused + residual)
            
        # Collapse sequence dimension for output
        # Temporal pooling
        # τ: ℝ^{B×L×h} → ℝ^{B×h}
        fused = fused.mean(dim=1)
            
        return fused


class ModalityGating(AbstractFusionBlock):
    """ Implementation of modality specific gating mechanism """

    # Input and hidden dimension definition
    # Input spaces 
    # X_m ∈ ℝ^{d_m} 
    # Hidden space 
    # H ∈ ℝ^h
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
        
        # Define gate networks for each modality - functions
        # g_m: ℝ^{d_m} → [0,1]
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

                # Compute gating activation 
                # α_m = σ(g_m(x_m))
                flat_feat = feat.reshape(-1, feat_dim)                
                gate = self.gate_networks[name](flat_feat)                
                gate = gate.reshape(batch_size, seq_len, 1)
                
                # Apply multiplicative gating 
                # y_m = x_m ⊙ α_m
                gated_features[name] = feat * gate
                
        return gated_features
