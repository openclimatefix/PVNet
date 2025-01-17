# test_dynamic_encoder.py


""" Testing for dynamic fusion encoder components """


import pytest
import torch
from typing import Dict

from pvnet.models.multimodal.encoders.dynamic_encoder import DynamicFusionEncoder


# Fixtures
@pytest.fixture
def minimal_config():
    """ Generate minimal config - basic functionality testing """
    sequence_length = 12
    hidden_dim = 48
    feature_dim = hidden_dim // sequence_length
    
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
                'out_features': hidden_dim,
                'number_of_conv3d_layers': 2,
                'conv3d_channels': 16,
                'batch_norm': True,
                'fc_dropout': 0.1
            },
            'pv': {
                'num_sites': 10,
                'out_features': feature_dim
            }
        },
        'cross_attention': {
            'embed_dim': feature_dim,
            'num_heads': 4,
            'dropout': 0.1,
            'num_modalities': 2
        },
        'modality_gating': {
            'feature_dims': {
                'sat': feature_dim,
                'pv': feature_dim
            },
            'hidden_dim': feature_dim,  # Changed to feature_dim
            'dropout': 0.1
        },
        'dynamic_fusion': {
            'feature_dims': {
                'sat': feature_dim,
                'pv': feature_dim
            },
            'hidden_dim': feature_dim,  # Changed to feature_dim
            'num_heads': 4,
            'dropout': 0.1,
            'fusion_method': 'weighted_sum',
            'use_residual': True
        }
    }


@pytest.fixture
def minimal_inputs(minimal_config):
    """ Generate minimal inputs with expected tensor shapes """

    batch_size = 2
    sequence_length = minimal_config['sequence_length']
    
    return {
        'sat': torch.randn(batch_size, 2, sequence_length, 24, 24),
        'pv': torch.randn(batch_size, sequence_length, 10)
    }


def create_encoder(config):
    """ Helper function - create encoder with consistent config """

    return DynamicFusionEncoder(
        sequence_length=config['sequence_length'],
        image_size_pixels=config['image_size_pixels'],
        modality_channels=config['modality_channels'],
        out_features=config['out_features'],
        modality_encoders=config['modality_encoders'],
        cross_attention=config['cross_attention'],
        modality_gating=config['modality_gating'],
        dynamic_fusion=config['dynamic_fusion'],
        hidden_dim=config['hidden_dim'],
        fc_features=config['fc_features']
    )


def test_initialisation(minimal_config):
    """ Verify encoder initialisation / component structure """

    encoder = create_encoder(minimal_config)
    assert isinstance(encoder, DynamicFusionEncoder)
    assert len(encoder.modality_encoders) == 2
    assert 'sat' in encoder.modality_encoders
    assert 'pv' in encoder.modality_encoders


def test_basic_forward(minimal_config, minimal_inputs):
    """ Test basic forward pass shape and values """

    encoder = create_encoder(minimal_config)
    
    with torch.no_grad():
        output = encoder(minimal_inputs)
    
    assert output.shape == (2, minimal_config['out_features'])
    assert not torch.isnan(output).any()
    assert output.dtype == torch.float32


# Modality handling tests
def test_single_modality(minimal_config, minimal_inputs):
    """ Test forward pass with single modality """
    encoder = create_encoder(minimal_config)
    
    # Test with only satellite data
    with torch.no_grad():
        sat_only = {'sat': minimal_inputs['sat']}
        output_sat = encoder(sat_only)
    
    assert output_sat.shape == (2, minimal_config['out_features'])
    assert not torch.isnan(output_sat).any()
    
    # Test with only PV data
    with torch.no_grad():
        pv_only = {'pv': minimal_inputs['pv']}
        output_pv = encoder(pv_only)
    
    assert output_pv.shape == (2, minimal_config['out_features'])
    assert not torch.isnan(output_pv).any()


def test_intermediate_shapes(minimal_config, minimal_inputs):
    """ Verify shapes of intermediate tensors throughout network """

    encoder = create_encoder(minimal_config)
    batch_size = minimal_inputs['sat'].size(0)
    sequence_length = minimal_config['sequence_length']
    feature_dim = minimal_config['hidden_dim'] // sequence_length
    
    def hook_fn(module, input, output):
        if isinstance(output, dict):
            for key, value in output.items():
                assert len(value.shape) == 3  # [batch, sequence, features]
                assert value.size(0) == batch_size
                assert value.size(1) == sequence_length
                assert value.size(2) == feature_dim
        elif isinstance(output, torch.Tensor):
            if len(output.shape) == 3:
                assert output.size(0) == batch_size
                assert output.size(1) == sequence_length
        
    # Register hooks
    if hasattr(encoder, 'gating'):
        encoder.gating.register_forward_hook(hook_fn)
    if hasattr(encoder, 'cross_attention'):
        encoder.cross_attention.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        encoder(minimal_inputs)


# Robustness tests
@pytest.mark.parametrize("batch_size", [1, 4])
def test_batch_sizes(minimal_config, minimal_inputs, batch_size):
    """ Test encoder behavior with different batch sizes """
    encoder = create_encoder(minimal_config)
    
    # Adjust input batch sizes
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


# Error handling tests
def test_empty_input(minimal_config):
    """ Verify error handling for empty input dictionary """

    encoder = create_encoder(minimal_config)
    with pytest.raises(ValueError, match="No valid features after encoding"):
        encoder({})


def test_invalid_modality(minimal_config, minimal_inputs):
    """ Verify error handling for invalid modality name """

    encoder = create_encoder(minimal_config)
    invalid_inputs = {'invalid_modality': minimal_inputs['sat']}
    with pytest.raises(ValueError):
        encoder(invalid_inputs)


def test_none_inputs(minimal_config, minimal_inputs):
    """ Test handling of None inputs for modalities """

    encoder = create_encoder(minimal_config)
    none_inputs = {'sat': None, 'pv': minimal_inputs['pv']}
    output = encoder(none_inputs)
    assert output.shape == (2, minimal_config['out_features'])


# Config tests
@pytest.mark.parametrize("sequence_length", [6, 24])
def test_variable_sequence_length(minimal_config, sequence_length):
    """Test different sequence lengths"""
    config = minimal_config.copy()
    config['sequence_length'] = sequence_length
    config['hidden_dim'] = sequence_length * 5
    
    encoder = create_encoder(config)
    batch_size = 2
    inputs = {
        'sat': torch.randn(batch_size, 2, sequence_length, 24, 24),
        'pv': torch.randn(batch_size, sequence_length, 10)
    }
    
    output = encoder(inputs)
    assert output.shape == (batch_size, config['out_features'])


# Architecture tests
def test_architecture_components(minimal_config):
    """Test specific architectural components and their connections"""

    encoder = create_encoder(minimal_config)
    
    # Test encoder layers
    assert hasattr(encoder, 'modality_encoders')
    assert hasattr(encoder, 'feature_projections')
    assert hasattr(encoder, 'fusion_module')
    assert hasattr(encoder, 'final_block')
        
    # Verify encoder has correct number of modalities
    assert len(encoder.modality_encoders) == len(minimal_config['modality_channels'])


def test_tensor_shape_tracking(minimal_config, minimal_inputs):
    """ Track tensor shapes through network layers """

    encoder = create_encoder(minimal_config)
    shapes = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            shapes[name] = output.shape if isinstance(output, torch.Tensor) else \
                          {k: v.shape for k, v in output.items()}
        return hook
    
    # Register shape tracking hooks
    encoder.modality_encoders['sat'].register_forward_hook(hook_fn('sat_encoder'))
    encoder.feature_projections['sat'].register_forward_hook(hook_fn('sat_projection'))
    encoder.fusion_module.register_forward_hook(hook_fn('fusion'))
    
    with torch.no_grad():
        output = encoder(minimal_inputs)
    
    # Verify expected shapes
    assert shapes['fusion'][1] == encoder.feature_dim
    assert output.shape[1] == minimal_config['out_features']


def test_modality_interactions(minimal_config, minimal_inputs):
    """ Test interaction between different modality combinations """

    encoder = create_encoder(minimal_config)
    batch_size = 2
    
    # Test different modality combinations
    test_cases = [
        ({'sat': minimal_inputs['sat']}, "single_sat"),
        ({'pv': minimal_inputs['pv']}, "single_pv"),
        (minimal_inputs, "both")
    ]
    
    outputs = {}
    for inputs, case_name in test_cases:
        with torch.no_grad():
            outputs[case_name] = encoder(inputs)
    
    # Verify outputs differ across modality combinations
    assert not torch.allclose(outputs['single_sat'], outputs['both'])
    assert not torch.allclose(outputs['single_pv'], outputs['both'])


def test_attention_behavior(minimal_config, minimal_inputs):
    """ Verify attention mechanism properties """

    encoder = create_encoder(minimal_config)
    attention_outputs = {}
    
    def attention_hook(module, input, output):
        if isinstance(output, dict):
            attention_outputs.update(output)
    
    if encoder.use_cross_attention:
        encoder.cross_attention.register_forward_hook(attention_hook)
    
    with torch.no_grad():
        encoder(minimal_inputs)
        
    if attention_outputs:
        # Verify attention weight distribution
        for modality, features in attention_outputs.items():
            std = features.std()
            assert std > 1e-6, "Attention weights too uniform"


@pytest.mark.parametrize("noise_level", [0.1, 0.5, 1.0])
def test_input_noise_robustness(minimal_config, minimal_inputs, noise_level):
    """ Test encoder stability under different noise levels """

    encoder = create_encoder(minimal_config)
    
    # Add noise to inputs
    noisy_inputs = {
        k: v + noise_level * torch.randn_like(v)
        for k, v in minimal_inputs.items()
    }
    
    with torch.no_grad():
        clean_output = encoder(minimal_inputs)
        noisy_output = encoder(noisy_inputs)
    
    # Check output stability
    relative_diff = (clean_output - noisy_output).abs().mean() / clean_output.abs().mean()
    assert not torch.isnan(relative_diff)
    assert not torch.isinf(relative_diff)
    assert relative_diff < noise_level * 10
