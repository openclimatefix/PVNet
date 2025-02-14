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
import logging

from pvnet.models.multimodal.attention_blocks import MultiheadAttention


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('fusion_blocks')


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
        logger.info(f"Initialising DynamicFusionModule with hidden_dim={hidden_dim}, num_heads={num_heads}")
        logger.debug(f"Feature dimensions: {feature_dims}")

        if hidden_dim <= 0 or num_heads <= 0:
            error_msg = f"Invalid dimensions: hidden_dim={hidden_dim}, num_heads={num_heads}"
            logger.error(error_msg)
            raise ValueError("hidden_dim and num_heads must be positive")
            
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        self.fusion_method = fusion_method
        self.use_residual = use_residual
        
        if fusion_method not in ["weighted_sum", "concat"]:
            error_msg = f"Invalid fusion method: {fusion_method}"
            logger.error(error_msg)
            raise ValueError(f"Invalid fusion method: {fusion_method}")
        
        # Projections - modality specific
        # φ_m: ℝ^{d_m} → ℝ^h for m ∈ M
        logger.debug(f"Creating projections for {len(feature_dims)} modalities")
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
        logger.debug("Initialising cross attention module")
        self.cross_attention = MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Weight computation network definition - dynamic
        # W: ℝ^h → [0,1]
        logger.debug("Creating weight computation network")
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
            logger.debug("Creating concat projection layers")
            self.output_projection = nn.Sequential(
                nn.Linear(hidden_dim * len(feature_dims), hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        if use_residual:
            logger.debug("Initialising layer normalization for residual connection")
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def _validate_features(self, features: Dict[str, torch.Tensor]) -> None:
        """ Validates input feature dimensions and sequence lengths """
        logger.debug("Starting feature validation")

        # Validate feature space dimensionality d_m
        # Validate sequence length L
        if not isinstance(features, dict) or not features:
            if isinstance(features, torch.Tensor):
                logger.debug("Skipping validation for single tensor input")
                return
            logger.error("Invalid features input: empty or not a dictionary")
            raise ValueError("Empty features dict")
        
        # Validate temporal dimensions L_m across modalities
        multi_dim_features = {}
        for name, feat in features.items():
            if feat is None:
                logger.error(f"None tensor found for modality: {name}")
                raise ValueError(f"None tensor for modality: {name}")
            
            if feat.ndim > 1:
                multi_dim_features[name] = feat.size(1)
                logger.debug(f"Modality {name} sequence length: {feat.size(1)}")

        # Verification step 
        # L_i = L_j ∀i,j ∈ M
        feature_lengths = set(multi_dim_features.values())
        if len(feature_lengths) > 1:
            logger.error(f"Inconsistent sequence lengths detected: {multi_dim_features}")
            raise ValueError(f"All modalities must have same sequence length. Current lengths: {multi_dim_features}")

    def compute_modality_weights(
        self,
        features: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        logger.debug(f"Computing modality weights for tensor of shape {features.shape}")

        """ Computation of attention weights for each feature """

        batch_size, seq_len = features.size(0), features.size(1)
        flat_features = features.reshape(-1, features.size(-1))
        weights = self.weight_network(flat_features)
        weights = weights.reshape(batch_size, seq_len, 1)
        
        if modality_mask is not None:
            logger.debug(f"Applying modality mask with shape {modality_mask.shape}")
            weights = weights.reshape(batch_size, -1, 1)[:, :modality_mask.size(1), :]
            weights = weights.masked_fill(~modality_mask.unsqueeze(-1), 0.0)
            
        # Normalisation of weights
        # α_m = w_m / Σ_j w_j
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)
        logger.debug(f"Computed weights shape: {weights.shape}")
        return weights

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        logger.info("Starting DynamicFusionModule forward pass")

        """ Forward pass for dynamic fusion """

        self._validate_features(features)
        logger.debug(f"Input features: {[f'{k}: {v.shape}' for k, v in features.items()]}")

        batch_size = next(iter(features.values())).size(0)
        seq_len = next(iter(features.values())).size(1)
        logger.debug(f"Batch size: {batch_size}, Sequence length: {seq_len}")
        
        # Apply modality-specific embeddings 
        # φ_m(x_m)
        projected_features = []
        for name, feat in features.items():
            if self.feature_dims[name] > 0:
                logger.debug(f"Projecting {name} features of shape {feat.shape}")
                flat_feat = feat.reshape(-1, feat.size(-1))
                proj = self.projections[name](flat_feat)
                projected = proj.reshape(batch_size, seq_len, self.hidden_dim)
                projected_features.append(projected)
                
        if not projected_features:
            logger.error("No valid features after projection")
            raise ValueError("No valid features after projection")
            
        # Tensor product of embedded features 
        # ⊗_{m∈M} φ_m(x_m)
        feature_stack = torch.stack(projected_features, dim=1)
        logger.debug(f"Feature stack shape after projection: {feature_stack.shape}")

        # Apply cross attention
        # A(⊗_{m∈M} φ_m(x_m))
        attended_features = []
        for i in range(feature_stack.size(1)):
            query = feature_stack[:, i]
            if feature_stack.size(1) > 1:
                logger.debug(f"Applying cross-attention for modality {i}")

                # Case |M| > 1
                # Apply cross-modal attention A_c
                key_value = feature_stack[:, [j for j in range(feature_stack.size(1)) if j != i]]
                attended = self.cross_attention(query, key_value.reshape(-1, seq_len, self.hidden_dim), 
                                            key_value.reshape(-1, seq_len, self.hidden_dim))
            else:
                logger.debug("Applying self-attention (single modality)")

                # Case |M| = 1
                # Apply self-attention A_s
                attended = self.cross_attention(query, query, query)
            attended_features.append(attended)
    
        # Compute mean representation
        # μ = 1/|M| Σ_{m∈M} A_m
        attended_features = torch.stack(attended_features, dim=1)
        attended_avg = attended_features.mean(dim=1)
        logger.debug(f"Attended features shape: {attended_features.shape}")
        
        # Apply attention mask 
        # M ∈ {0,1}^{B×L}
        if modality_mask is not None:
            logger.debug(f"Applying modality mask of shape {modality_mask.shape}")
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
        logger.debug(f"Weighted features shape: {weighted_features.shape}")
        
        if self.fusion_method == "weighted_sum":
            logger.debug("Applying weighted sum fusion")
            fused = weighted_features.sum(dim=1)
        else:
            logger.debug("Applying concat fusion")
            concat = weighted_features.reshape(batch_size, seq_len, -1)
            fused = self.output_projection(concat)
            
        # Application of residual
        # r(x) = LayerNorm(x + μ(x))
        if self.use_residual:
            logger.debug("Applying residual connection")
            residual = feature_stack.mean(dim=1)
            fused = self.layer_norm(fused + residual)
            
        # Collapse sequence dimension for output
        # Temporal pooling
        # τ: ℝ^{B×L×h} → ℝ^{B×h}
        fused = fused.mean(dim=1)
        logger.debug(f"Final fused output shape: {fused.shape}")
            
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
        logger.info(f"Initialising ModalityGating with hidden_dim={hidden_dim}")
        logger.debug(f"Feature dimensions: {feature_dims}")

        if hidden_dim <= 0:
            logger.error(f"Invalid hidden_dim: {hidden_dim}")
            raise ValueError("hidden_dim must be positive")

        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        
        # Define gate networks for each modality - functions
        # g_m: ℝ^{d_m} → [0,1]
        logger.debug("Creating gate networks")
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
        logger.debug("Validating input features")

        if not features:
            logger.error("Empty features dictionary provided")
            raise ValueError("Empty features dict")

        for name, feat in features.items():
            if feat is None:
                logger.error(f"None tensor found for modality: {name}")
                raise ValueError(f"None tensor for modality: {name}")
            logger.debug(f"Feature {name} shape: {feat.shape}")

    def forward(
        self,
        features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        logger.info("Starting ModalityGating forward pass")
        logger.debug(f"Input features: {[f'{k}: {v.shape}' for k, v in features.items()]}")

        """ Application of modality specific gating """

        self._validate_features(features)
        gated_features = {}
        
        for name, feat in features.items():
            if feat is not None and name in self.gate_networks:
                batch_size, seq_len, feat_dim = feat.shape
                logger.debug(f"Processing {name} modality with shape {feat.shape}")

                # Compute gating activation 
                # α_m = σ(g_m(x_m))
                flat_feat = feat.reshape(-1, feat_dim)                
                gate = self.gate_networks[name](flat_feat)                
                gate = gate.reshape(batch_size, seq_len, 1)
                logger.debug(f"Computed gate values for {name} with shape {gate.shape}")

                # Apply multiplicative gating 
                # y_m = x_m ⊙ α_m
                gated_features[name] = feat * gate
                logger.debug(f"Applied gating to {name}, output shape: {gated_features[name].shape}")
                
        return gated_features
