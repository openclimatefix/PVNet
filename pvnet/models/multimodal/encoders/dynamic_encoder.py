# dynamic_encoder.py

""" 
Dynamic fusion encoder implementation for multimodal learning 

Defines PVEncoder, DynamicFusionEncoder and DynamicResidualEncoder
"""

from typing import Dict, Optional, List, Union
import torch
from torch import nn

from pvnet.models.multimodal.encoders.basic_blocks import AbstractNWPSatelliteEncoder
from pvnet.models.multimodal.fusion_blocks import DynamicFusionModule, ModalityGating
from pvnet.models.multimodal.attention_blocks import CrossModalAttention, SelfAttention
from pvnet.models.multimodal.encoders.encoders3d import DefaultPVNet2


# Attention head compatibility function
def get_compatible_heads(dim: int, target_heads: int) -> int:
    """ Calculate largest compatible number of heads <= target_heads """

    for h in range(min(target_heads, dim), 0, -1):
        if dim % h == 0:
            return h
    return 1


# Processes PV data maintaining temporal sequence
class PVEncoder(nn.Module):
    """ PV specific encoder implementation with sequence preservation """

    def __init__(self, sequence_length: int, num_sites: int, out_features: int):
        super().__init__()
        self.sequence_length = sequence_length
        self.num_sites = num_sites
        self.out_features = out_features
        
        # Basic feature extraction network
        self.encoder = nn.Sequential(
            nn.Linear(num_sites, out_features),
            nn.LayerNorm(out_features),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):

        # Sequential processing - maintain temporal order
        batch_size = x.shape[0]
        out = []
        for t in range(self.sequence_length):
            out.append(self.encoder(x[:, t]))\
        # Reshape maintaining sequence dimension
        return torch.stack(out, dim=1)


# Primary fusion encoder implementation
class DynamicFusionEncoder(AbstractNWPSatelliteEncoder):
    def __init__(
        self,
        sequence_length: int,
        image_size_pixels: int,
        modality_channels: Dict[str, int],
        out_features: int,
        modality_encoders: Dict[str, dict],
        cross_attention: Dict,
        modality_gating: Dict,
        dynamic_fusion: Dict,
        hidden_dim: int = 256,
        fc_features: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_gating: bool = True,
        use_cross_attention: bool = True
    ):
        """ Dynamic fusion encoder initialisation """

        super().__init__(
            sequence_length=sequence_length,
            image_size_pixels=image_size_pixels,
            in_channels=sum(modality_channels.values()),
            out_features=out_features
        )

        # Dimension validation and compatibility
        if hidden_dim % sequence_length != 0:
            feature_dim = ((hidden_dim + sequence_length - 1) // sequence_length)
            hidden_dim = feature_dim * sequence_length
        else:
            feature_dim = hidden_dim // sequence_length

        # Attention mechanism setup
        attention_heads = cross_attention.get('num_heads', num_heads)
        attention_heads = get_compatible_heads(feature_dim, attention_heads)
        
        # Feature dimension adjustment for attention
        if feature_dim < attention_heads:
            feature_dim = attention_heads
            hidden_dim = feature_dim * sequence_length
        elif feature_dim % attention_heads != 0:
            feature_dim = ((feature_dim + attention_heads - 1) // attention_heads) * attention_heads
            hidden_dim = feature_dim * sequence_length

        # Architecture dimensions
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.modalities = list(modality_channels.keys())
        
        # Update configs with validated dimensions
        cross_attention['num_heads'] = attention_heads
        dynamic_fusion['num_heads'] = attention_heads
            
        # Modality specific encoder instantiation
        self.modality_encoders = nn.ModuleDict()
        for modality, config in modality_encoders.items():
            config = config.copy()
            if 'nwp' in modality or 'sat' in modality:

                # Image based modality encoder
                encoder = DefaultPVNet2(
                    sequence_length=sequence_length,
                    image_size_pixels=config.get('image_size_pixels', image_size_pixels),
                    in_channels=modality_channels[modality],
                    out_features=hidden_dim,
                    number_of_conv3d_layers=config.get('number_of_conv3d_layers', 4),
                    conv3d_channels=config.get('conv3d_channels', 32),
                    batch_norm=config.get('batch_norm', True),
                    fc_dropout=dropout
                )
                
                self.modality_encoders[modality] = nn.Sequential(
                    encoder,
                    nn.Linear(hidden_dim, sequence_length * feature_dim),
                    nn.Unflatten(-1, (sequence_length, feature_dim))
                )
            elif modality == 'pv':

                # PV specific encoder
                self.modality_encoders[modality] = PVEncoder(
                    sequence_length=sequence_length,
                    num_sites=config['num_sites'],
                    out_features=feature_dim
                )
        
        # Feature transformation layers
        self.feature_projections = nn.ModuleDict({
            modality: nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for modality in modality_channels.keys()
        })
        
        # Modality gating mechanism
        self.use_gating = use_gating
        if use_gating:
            gating_config = modality_gating.copy()
            gating_config.update({
                'feature_dims': {mod: feature_dim for mod in modality_channels.keys()},
                'hidden_dim': feature_dim
            })
            self.gating = ModalityGating(**gating_config)
            
        # Cross modal attention mechanism
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            attention_config = cross_attention.copy()
            attention_config.update({
                'embed_dim': feature_dim,
                'num_heads': attention_heads,
                'dropout': dropout
            })
            self.cross_attention = CrossModalAttention(**attention_config)
            
        # Dynamic fusion implementation
        fusion_config = dynamic_fusion.copy()
        fusion_config.update({
            'feature_dims': {mod: feature_dim for mod in modality_channels.keys()},
            'hidden_dim': feature_dim,
            'num_heads': attention_heads,
            'dropout': dropout
        })
        self.fusion_module = DynamicFusionModule(**fusion_config)
        
        # Output network definition
        self.final_block = nn.Sequential(
            nn.Linear(hidden_dim, fc_features),
            nn.LayerNorm(fc_features),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_features, out_features),
            nn.ELU(),
        )
        
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        """ Dynamic fusion forward pass implementation """

        encoded_features = {}

        # Modality specific encoding
        for modality, x in inputs.items():
            if modality not in self.modality_encoders or x is None:
                continue
                
            # Feature extraction and projection
            encoded = self.modality_encoders[modality](x)
            projected = torch.stack([
                self.feature_projections[modality](encoded[:, t])
                for t in range(self.sequence_length)
            ], dim=1)
            
            encoded_features[modality] = projected
            
        if not encoded_features:
            raise ValueError("No valid features after encoding")
            
        # Apply modality interaction mechanisms
        if self.use_gating:
            encoded_features = self.gating(encoded_features)
            
        if self.use_cross_attention and len(encoded_features) > 1:
            encoded_features = self.cross_attention(encoded_features, mask)
            
        # Feature fusion and output generation
        fused_features = self.fusion_module(encoded_features, mask)
        batch_size = fused_features.size(0)
        fused_features = fused_features.repeat(1, self.sequence_length)  
        output = self.final_block(fused_features)
        
        return output


class DynamicResidualEncoder(DynamicFusionEncoder):
    """ Dynamic fusion implementation with residual connectivity """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Enhanced projection with residual pathways
        self.feature_projections = nn.ModuleDict({
            modality: nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(kwargs.get('dropout', 0.1)),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
            )
            for modality in kwargs['modality_channels'].keys()
        })

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        """ Forward implementation with residual pathways """

        encoded_features = {}

        # Feature extraction with residual connections
        for modality, x in inputs.items():
            if modality not in self.modality_encoders or x is None:
                continue
                
            encoded = self.modality_encoders[modality](x)            
            projected = encoded + self.feature_projections[modality](encoded)
            encoded_features[modality] = projected
            
        if not encoded_features:
            raise ValueError("No valid features after encoding")
            
        # Gating with residual pathways
        if self.use_gating:
            gated_features = self.gating(encoded_features)
            for modality in encoded_features:
                gated_features[modality] = gated_features[modality] + encoded_features[modality]
            encoded_features = gated_features
            
        # Attention with residual pathways
        if self.use_cross_attention and len(encoded_features) > 1:
            attended_features = self.cross_attention(encoded_features, mask)
            for modality in encoded_features:
                attended_features[modality] = attended_features[modality] + encoded_features[modality]
            encoded_features = attended_features
            
        # Final fusion and output generation
        fused_features = self.fusion_module(encoded_features, mask)        
        fused_features = fused_features.repeat(1, self.sequence_length)
        output = self.final_block(fused_features)
        
        return output
