import pytest
import torch
from typing import Dict

from pvnet.models.multimodal.encoders.dynamic_encoder import DynamicFusionEncoder

@pytest.fixture
def minimal_config():
    """Minimal configuration for testing basic functionality"""
    sequence_length = 12
    hidden_dim = 60  # Chosen so it divides evenly by sequence_length (60/12 = 5)
    
    # Important: feature_dim needs to match between modalities
    feature_dim = hidden_dim // sequence_length  # This is 5
    
    return {
        'sequence_length': sequence_length,
        'image_size_pixels': 24,
        'modality_channels': {
            'sat': 2,
            'pv': 10
        },
        'out_features': 32,
        'hidden_dim': hidden_dim,
        'fc_features': 32,
        'modality_encoders': {
            'sat': {
                'image_size_pixels': 24,
                'out_features': feature_dim * sequence_length,  # 60
                'number_of_conv3d_layers': 2,
                'conv3d_channels': 16,
                'batch_norm': True,
                'fc_dropout': 0.1
            },
            'pv': {
                'num_sites': 10,
                'out_features': feature_dim  # 5 - this ensures proper dimension
            }
        },
        'cross_attention': {
            'embed_dim': hidden_dim,
            'num_heads': 4,
            'dropout': 0.1,
            'num_modalities': 2
        },
        'modality_gating': {
            'feature_dims': {
                'sat': hidden_dim,
                'pv': hidden_dim
            },
            'hidden_dim': hidden_dim,
            'dropout': 0.1
        },
        'dynamic_fusion': {
            'feature_dims': {
                'sat': hidden_dim,
                'pv': hidden_dim
            },
            'hidden_dim': hidden_dim,
            'num_heads': 4,
            'dropout': 0.1,
            'fusion_method': 'weighted_sum',
            'use_residual': True
        }
    }

@pytest.fixture
def minimal_inputs(minimal_config):
    """Generate minimal test inputs"""
    batch_size = 2
    sequence_length = minimal_config['sequence_length']
    
    return {
        'sat': torch.randn(batch_size, 2, sequence_length, 24, 24),
        'pv': torch.randn(batch_size, sequence_length, 10)
    }

def test_batch_sizes(self, minimal_config, minimal_inputs, batch_size):
    """Test different batch sizes"""
    encoder = DynamicFusionEncoder(
        sequence_length=minimal_config['sequence_length'],
        image_size_pixels=minimal_config['image_size_pixels'],
        modality_channels=minimal_config['modality_channels'],
        out_features=minimal_config['out_features'],
        modality_encoders=minimal_config['modality_encoders'],
        cross_attention=minimal_config['cross_attention'],
        modality_gating=minimal_config['modality_gating'],
        dynamic_fusion=minimal_config['dynamic_fusion'],
        hidden_dim=minimal_config['hidden_dim'],
        fc_features=minimal_config['fc_features']
    )
    
    # Adjust input batch sizes - fixed repeat logic
    adjusted_inputs = {}
    for k, v in minimal_inputs.items():
        if batch_size < v.size(0):
            adjusted_inputs[k] = v[:batch_size]
        else:
            repeat_factor = batch_size // v.size(0)
            adjusted_inputs[k] = v.repeat(repeat_factor, *[1]*(len(v.shape)-1))
    
    with torch.no_grad():
        output = encoder(adjusted_inputs)
    
    assert output.shape == (batch_size, minimal_config['out_features'])
    assert not torch.isnan(output).any()