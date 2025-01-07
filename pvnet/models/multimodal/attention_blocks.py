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


class AbstractAttentionBlock(nn.Module, ABC):
    """ Abstract attention base class definition """

    # Forward pass
    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        pass


# Splits input into multiple heads - scales attention scores for stability
class MultiheadAttention(AbstractAttentionBlock):
    """ Multihead attention implementation / definition """

    # Initialisation of multihead attention 
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):

        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim not divisible by num_heads")
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
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
        
        # Projection and reshape - define attention scores
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Attention weights and subsequent output
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        return self.out_proj(attn_output)


# Enables singular modality to 'attend' to others utilising specific attention block
class CrossModalAttention(AbstractAttentionBlock):
    """ CrossModal attention - interaction between multiple modalities """

    # Initialisation of CrossModal attention 
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        num_modalities: int = 2
    ):

        super().__init__()
        self.num_modalities = num_modalities
        self.attention_blocks = nn.ModuleList([
            MultiheadAttention(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_modalities)
        ])
        self.dropout = nn.Dropout(dropout)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_modalities)])

    # Forward pass - CrossModal attention    
    def forward(
        self,
        modalities: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:

        updated_modalities = {}
        modality_keys = list(modalities.keys())

        for i, key in enumerate(modality_keys):
            query = modalities[key]
            # Combine other modalities as key-value pairs
            other_modalities = [modalities[k] for k in modality_keys if k != key]
            if other_modalities:
                key_value = torch.cat(other_modalities, dim=1)
                
                # Apply attention block for this modality
                attn_output = self.attention_blocks[i](query, key_value, key_value, mask)
                attn_output = self.dropout(attn_output)
                updated_modalities[key] = self.layer_norms[i](query + attn_output)
            else:
                # If no other modalities, pass through
                updated_modalities[key] = query

        return updated_modalities


# Permits each element in input sequence to attend all other elements
class SelfAttention(AbstractAttentionBlock):
    """ SelfAttention block for singular modality """

    # Initialisation of self attention 
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):

        super().__init__()
        self.attention = MultiheadAttention(embed_dim, num_heads, dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    # Forward pass - self attention
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        attn_output = self.attention(x, x, x, mask)
        attn_output = self.dropout(attn_output)
        return self.layer_norm(x + attn_output)
        