# attention_blocks.py

""" 
Attention blocks for multimodal dynamic fusion implementation

Fundamentally a foundation script, that defines the mechanisms MultiheadAttention, CrossModalAttention, and SelfAttention

Aformentioned attention blocks enable early cross modal interaction, with permits each modality to learn from features of other modalities as opposed to independent processing
"""


import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Optional
from torch import nn
import logging


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('attention_blocks')


class AbstractAttentionBlock(nn.Module, ABC):
    """ Abstract attention base class definition - for all derived attention mechanisms """

    # Forward pass
    # f: X → Y - abstract attention space
    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        pass


# Partitions input into h parallel heads with scaled dot-product scoring
# s(x) = <q,k>/√d_k
class MultiheadAttention(AbstractAttentionBlock):
    """
    Multihead attention implementation / definition

    Scaled dot-product attention
    
    Parallel attention heads permit model to jointly 'attend' information from different representation subspaces
    """

    # Initialisation of h parallel attention mechanisms 
    # A_i: ℝ^d → ℝ^{d/h}
    # Definition of embedding dimension d ∈ ℕ 
    # Definition of attention heads h | d mod h = 0
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()

        logger.info(f"Initialising MultiheadAttention with embed_dim={embed_dim}, num_heads={num_heads}")
        
        if embed_dim % num_heads != 0:
            error_msg = f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        logger.debug(f"Head dimension: {self.head_dim}, Scale factor: {self.scale}")

        # Linear transformations for query-key-value projections
        # W_Q, W_K, W_V ∈ ℝ^{d×d} 
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    # Forward pass - multihead attention    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        batch_size = query.shape[0]
        logger.debug(f"Input shapes - Query: {query.shape}, Key: {key.shape}, Value: {value.shape}")

        # Transform and partition input tensor
        # ℝ^{B×L×d} → ℝ^{B×h×L×(d/h)}
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)        

        logger.debug(f"After projection shapes - Q: {q.shape}, K: {k.shape}, V: {v.shape}")

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        logger.debug(f"Attention scores shape: {scores.shape}")
    
        if mask is not None:
            logger.debug(f"Applying mask with shape: {mask.shape}")
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Compute attention distribution
        # α = softmax(QK^T/√d_k)
        # Compute weighted context 
        # Σ_i α_i v_i
        attn_weights = F.softmax(scores, dim=-1)
        logger.debug(f"Attention weights shape: {attn_weights.shape}")

        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        logger.debug(f"Attention output shape (before reshape): {attn_output.shape}")

        # Restore tensor dimensionality
        # ℝ^{B×h×L×(d/h)} → ℝ^{B×L×d}
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        logger.debug(f"Final output shape: {attn_output.shape}")

        return self.out_proj(attn_output)


# Enables singular modality to 'attend' to others context utilising specific attention block
class CrossModalAttention(AbstractAttentionBlock):
    """ CrossModal attention - interaction between multiple modalities """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        num_modalities: int = 2
    ):
        super().__init__()
        logger.info(f"Initialising CrossModalAttention with {num_modalities} modalities")

        self.num_modalities = num_modalities

        # Parallel attention mechanisms for M modalities
        # {A_i}_{i=1}^M 
        self.attention_blocks = nn.ModuleList([
            MultiheadAttention(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_modalities)
        ])
        self.dropout = nn.Dropout(dropout)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_modalities)])

    def forward(
        self,
        modalities: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        logger.info("Processing CrossModalAttention forward pass")
        logger.debug(f"Input modalities: {[f'{k}: {v.shape}' for k, v in modalities.items()]}")

        updated_modalities = {}
        modality_keys = list(modalities.keys())

        for i, key in enumerate(modality_keys):
            logger.debug(f"Processing modality: {key}")
            query = modalities[key]
            other_modalities = [modalities[k] for k in modality_keys if k != key]

            if other_modalities:
                # Concatenate context modalities
                # C = [m_1; ...; m_{i-1}; m_{i+1}; ...; m_M]
                logger.debug(f"Concatenating {len(other_modalities)} other modalities")
                key_value = torch.cat(other_modalities, dim=1)
                logger.debug(f"Concatenated key_value shape: {key_value.shape}")               
                attn_output = self.attention_blocks[i](query, key_value, key_value, mask)
            else:
                # Apply self-attention 
                # A(x,x,x) when |M| = 1
                logger.debug("No other modalities found, applying self-attention")
                attn_output = self.attention_blocks[i](query, query, query, mask)
                
            attn_output = self.dropout(attn_output)
            updated_modalities[key] = self.layer_norms[i](query + attn_output)
            logger.debug(f"Updated modality {key} shape: {updated_modalities[key].shape}")

        return updated_modalities


# Permits each element in input sequence to attend all other elements
# I.e. all pair interaction via self attention
# A(x_i, {x_j}_{j=1}^L)
class SelfAttention(AbstractAttentionBlock):
    """ SelfAttention block for singular modality """

    # Initialisation of h parallel self-attention mechanisms
    # S_i: ℝ^d → ℝ^{d/h}
    # Total embedding dimension and quantity of parallel attention heads
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        logger.info(f"Initialising SelfAttention with embed_dim={embed_dim}, num_heads={num_heads}")

        self.attention = MultiheadAttention(embed_dim, num_heads, dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    # Forward pass - self attention
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        logger.debug(f"SelfAttention input shape: {x.shape}")

        # Self-attention operation
        # SA(x) = LayerNorm(x + A(x,x,x))
        attn_output = self.attention(x, x, x, mask)
        logger.debug(f"SelfAttention output shape (pre-dropout): {attn_output.shape}")
        attn_output = self.dropout(attn_output)
        return self.layer_norm(x + attn_output)
        