# dynamic_encoder.py

""" Dynamic fusion encoder implementation for multimodal learning """


from typing import Dict, Optional, List, Union
import torch
from torch import nn

from pvnet.models.multimodal.encoders.basic_blocks import AbstractNWPSatelliteEncoder
from pvnet.models.multimodal.fusion_blocks import DynamicFusionModule, ModalityGating
from pvnet.models.multimodal.attention_blocks import CrossModalAttention, SelfAttention
from pvnet.models.multimodal.encoders.encoders3d import DefaultPVNet2


class PVEncoder(nn.Module):
    """ Simplified PV encoder - maintains sequence dimension """

    def __init__(self, sequence_length: int, num_sites: int, out_features: int):
        super().__init__()
        self.sequence_length = sequence_length
        self.num_sites = num_sites
        self.out_features = out_features
        
        # Process each timestep independently
        self.encoder = nn.Sequential(
            nn.Linear(num_sites, out_features),
            nn.LayerNorm(out_features),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        # x: [batch_size, sequence_length, num_sites]
        batch_size = x.shape[0]
        # Process each timestep
        out = []
        for t in range(self.sequence_length):
            out.append(self.encoder(x[:, t]))
        # Stack along sequence dimension
        return torch.stack(out, dim=1)  # [batch_size, sequence_length, out_features]


class DynamicFusionEncoder(AbstractNWPSatelliteEncoder):

    """Encoder that implements dynamic fusion of satellite/NWP data streams"""
    
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
        """Dynamic fusion encoder for multimodal satellite/NWP data."""
        super().__init__(
            sequence_length=sequence_length,
            image_size_pixels=image_size_pixels,
            in_channels=sum(modality_channels.values()),
            out_features=out_features
        )
        
        self.modalities = list(modality_channels.keys())
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        
        # Initialize modality-specific encoders
        self.modality_encoders = nn.ModuleDict()
        for modality, config in modality_encoders.items():
            config = config.copy()
            if 'nwp' in modality or 'sat' in modality:
                encoder = DefaultPVNet2(
                    sequence_length=sequence_length,
                    image_size_pixels=config.get('image_size_pixels', image_size_pixels),
                    in_channels=modality_channels[modality],
                    out_features=config.get('out_features', hidden_dim),
                    number_of_conv3d_layers=config.get('number_of_conv3d_layers', 4),
                    conv3d_channels=config.get('conv3d_channels', 32),
                    batch_norm=config.get('batch_norm', True),
                    fc_dropout=config.get('fc_dropout', 0.2)
                )

                self.modality_encoders[modality] = nn.Sequential(
                    encoder,
                    nn.Unflatten(1, (sequence_length, hidden_dim//sequence_length))  
                )

            elif modality == 'pv':
                self.modality_encoders[modality] = PVEncoder(
                    sequence_length=sequence_length,
                    num_sites=config['num_sites'],
                    out_features=hidden_dim
                )
        
        # Feature projections
        self.feature_projections = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for modality in modality_channels.keys()
        })
        
        # Optional modality gating
        self.use_gating = use_gating
        if use_gating:
            gating_config = modality_gating.copy()
            gating_config['feature_dims'] = {
                mod: hidden_dim for mod in modality_channels.keys()
            }
            self.gating = ModalityGating(**gating_config)
            
        # Optional cross-modal attention
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            attention_config = cross_attention.copy()
            attention_config['embed_dim'] = hidden_dim
            self.cross_attention = CrossModalAttention(**attention_config)
            
        # Dynamic fusion module
        fusion_config = dynamic_fusion.copy()
        fusion_config['feature_dims'] = {
            mod: hidden_dim for mod in modality_channels.keys()
        }
        fusion_config['hidden_dim'] = hidden_dim
        self.fusion_module = DynamicFusionModule(**fusion_config)
        
        # Final output projection
        self.final_block = nn.Sequential(
            nn.Linear(hidden_dim * sequence_length, fc_features),
            nn.ELU(),
            nn.Linear(fc_features, out_features),
            nn.ELU(),
        )
        
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the dynamic fusion encoder"""
        # Initial encoding of each modality
        encoded_features = {}
        for modality, x in inputs.items():
            if modality not in self.modality_encoders:
                continue
            
            # Apply modality-specific encoder
            # Output shape: [batch_size, sequence_length, hidden_dim]
            encoded_features[modality] = self.modality_encoders[modality](x)
            
        if not encoded_features:
            raise ValueError("No valid features found in inputs")
            
        # Apply modality gating if enabled
        if self.use_gating:
            encoded_features = self.gating(encoded_features)
            
        # Apply cross-modal attention if enabled and more than one modality
        if self.use_cross_attention and len(encoded_features) > 1:
            encoded_features = self.cross_attention(encoded_features, mask)
            
        # Apply dynamic fusion
        fused_features = self.fusion_module(encoded_features, mask)  # [batch, sequence, hidden]
        
        # Reshape and apply final projection
        batch_size = fused_features.size(0)
        fused_features = fused_features.reshape(batch_size, -1)  # Flatten sequence dimension
        output = self.final_block(fused_features)
        
        return output


class DynamicResidualEncoder(DynamicFusionEncoder):
    """Dynamic fusion encoder with residual connections"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Override feature projections to include residual connections
        self.feature_projections = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(kwargs.get('dropout', 0.1)),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim)
            )
            for modality in kwargs['modality_channels'].keys()
        })