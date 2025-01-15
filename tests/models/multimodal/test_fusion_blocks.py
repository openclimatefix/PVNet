# test_fusion_blocks.py

""" 
Tests fusion block components - verifying initialisation, forward pass, gradient flow, and fusion methods
"""

import torch
import pytest
from pvnet.models.multimodal.fusion_blocks import DynamicFusionModule, ModalityGating


@pytest.fixture
def config():
    """ Test configuration parameters """
    return {
        'feature_dims': {
            'visual': 64,
            'text': 32,
            'audio': 48
        },
        'hidden_dim': 128,
        'num_heads': 4,
        'batch_size': 8,
        'sequence_length': 12,
        'dropout': 0.1
    }


@pytest.fixture
def multimodal_features(config):
    """ Generate complete multimodal input features """
    batch_size = config['batch_size']
    seq_len = config['sequence_length']
    return {
        'visual': torch.randn(batch_size, seq_len, config['feature_dims']['visual']),
        'text': torch.randn(batch_size, seq_len, config['feature_dims']['text']),
        'audio': torch.randn(batch_size, seq_len, config['feature_dims']['audio'])
    }


@pytest.fixture
def attention_mask(config):
    """ Generate attention mask with correct shape """
    batch_size = config['batch_size']
    num_modalities = len(config['feature_dims'])
    return torch.ones(batch_size, num_modalities, dtype=torch.bool)


# DynamicFusionModule Tests
def test_dynamic_fusion_initialization(config):
    """ Verify initialization and parameter validation """
    # Test valid initialization
    fusion = DynamicFusionModule(
        feature_dims=config['feature_dims'],
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        dropout=config['dropout']
    )
    
    # Assert components
    assert len(fusion.projections) == len(config['feature_dims'])
    assert fusion.hidden_dim == config['hidden_dim']
    assert isinstance(fusion.cross_attention.embed_dim, int)
    assert isinstance(fusion.weight_network, torch.nn.Sequential)
    
    # Test invalid hidden_dim
    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        DynamicFusionModule(
            feature_dims=config['feature_dims'],
            hidden_dim=0
        )
    
    # Test invalid num_heads
    with pytest.raises(ValueError, match="num_heads must be positive"):
        DynamicFusionModule(
            feature_dims=config['feature_dims'],
            num_heads=0
        )


def test_dynamic_fusion_feature_validation(config, multimodal_features):
    """ Test feature validation """
    fusion = DynamicFusionModule(
        feature_dims=config['feature_dims'],
        hidden_dim=config['hidden_dim']
    )
    
    # Test empty features
    with pytest.raises(ValueError, match="Empty features dictionary"):
        fusion({})
    
    # Test None tensor
    invalid_features = multimodal_features.copy()
    invalid_features['visual'] = None
    with pytest.raises(ValueError, match="None tensor for modality"):
        fusion(invalid_features)


def test_dynamic_fusion_forward_weighted_sum(config, multimodal_features):
    """ Forward pass with weighted sum fusion """
    fusion = DynamicFusionModule(
        feature_dims=config['feature_dims'],
        hidden_dim=config['hidden_dim'],
        fusion_method="weighted_sum"
    )
    output = fusion(multimodal_features)
    
    # Assert output shape
    expected_shape = (config['batch_size'], config['hidden_dim'])
    assert output.shape == expected_shape
    assert not torch.isnan(output).any()


def test_dynamic_fusion_forward_concat(config, multimodal_features):
    """ Forward pass with concat fusion """
    fusion = DynamicFusionModule(
        feature_dims=config['feature_dims'],
        hidden_dim=config['hidden_dim'],
        fusion_method="concat"
    )
    output = fusion(multimodal_features)
    
    # Assert output shape
    expected_shape = (config['batch_size'], config['hidden_dim'])
    assert output.shape == expected_shape
    assert not torch.isnan(output).any()


def test_dynamic_fusion_masked_attention(config, multimodal_features, attention_mask):
    """ Test attention masking """
    fusion = DynamicFusionModule(
        feature_dims=config['feature_dims'],
        hidden_dim=config['hidden_dim']
    )
    
    attention_mask[0, 0] = False
    output = fusion(multimodal_features, attention_mask)
    
    # Assert output shape and values
    expected_shape = (config['batch_size'], config['hidden_dim'])
    assert output.shape == expected_shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_dynamic_fusion_gradient_flow(config, multimodal_features):
    """ Verify gradient computation """
    fusion = DynamicFusionModule(
        feature_dims=config['feature_dims'],
        hidden_dim=config['hidden_dim']
    )
    
    # Forward pass
    output = fusion(multimodal_features)
    loss = output.sum()
    loss.backward()
    
    # Assert gradients for key components
    for proj in fusion.projections.values():
        assert proj[0].weight.grad is not None
        assert not torch.isnan(proj[0].weight.grad).any()
    
    assert fusion.cross_attention.q_proj.weight.grad is not None
    assert fusion.weight_network[0].weight.grad is not None


def test_dynamic_fusion_different_sequence_lengths(config):
    """ Test with varying sequence lengths """
    fusion = DynamicFusionModule(
        feature_dims=config['feature_dims'],
        hidden_dim=config['hidden_dim']
    )
    
    varying_features = {
        'visual': torch.randn(config['batch_size'], 10, config['feature_dims']['visual']),
        'text': torch.randn(config['batch_size'], 15, config['feature_dims']['text']),
        'audio': torch.randn(config['batch_size'], 12, config['feature_dims']['audio'])
    }
    
    with pytest.raises(ValueError, match=r"All modalities must have the same sequence length"):
        output = fusion(varying_features)


# ModalityGating Tests
def test_modality_gating_initialization(config):
    """ Verify initialisation """
    gating = ModalityGating(
        feature_dims=config['feature_dims'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout']
    )
    
    assert len(gating.gate_networks) == len(config['feature_dims'])
    for name, network in gating.gate_networks.items():
        assert isinstance(network, torch.nn.Sequential)
        # Verify input dimension of first layer matches feature dimension
        assert network[0].in_features == config['feature_dims'][name]
        
    # Test invalid hidden_dim
    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        ModalityGating(
            feature_dims=config['feature_dims'],
            hidden_dim=0
        )


def test_modality_gating_forward(config, multimodal_features):
    """ Test forward pass """
    gating = ModalityGating(
        feature_dims=config['feature_dims'],
        hidden_dim=config['hidden_dim']
    )
    outputs = gating(multimodal_features)
    
    # Verify output shapes and properties
    assert len(outputs) == len(multimodal_features)
    for modality, output in outputs.items():
        assert output.shape == multimodal_features[modality].shape  # Should match 3D input shape
        assert len(output.shape) == 3  # Ensure 3D output (batch, sequence, features)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        # Verify gating values are between 0 and 1
        gates = output / (multimodal_features[modality] + 1e-8)  # Avoid division by zero
        assert torch.all((gates >= 0) & (gates <= 1 + 1e-6))


def test_modality_gating_gradient_flow(config, multimodal_features):
    """ Verify gradient computation """
    gating = ModalityGating(
        feature_dims=config['feature_dims'],
        hidden_dim=config['hidden_dim']
    )
    outputs = gating(multimodal_features)
    
    # Compute loss over all timesteps
    loss = sum(output.sum() for output in outputs.values())
    loss.backward()
    
    # Assert gradients
    for network in gating.gate_networks.values():
        assert network[0].weight.grad is not None
        assert not torch.isnan(network[0].weight.grad).any()


def test_modality_gating_partial_features(config, multimodal_features):
    """ Test with feature subset """
    partial_features = {
        'visual': multimodal_features['visual'],
        'text': multimodal_features['text']
    }
    gating = ModalityGating(
        feature_dims=config['feature_dims'],
        hidden_dim=config['hidden_dim']
    )
    outputs = gating(partial_features)
    
    assert len(outputs) == len(partial_features)
    assert 'audio' not in outputs
    # Verify 3D shapes are maintained
    for modality, output in outputs.items():
        assert len(output.shape) == 3
        assert output.shape == partial_features[modality].shape


def test_modality_gating_edge_cases(config):
    """ Test edge cases """
    gating = ModalityGating(feature_dims={'visual': 64})
    
    # Empty input validation
    with pytest.raises(ValueError, match="Empty features dictionary"):
        gating({})
    
    # Test with single timestep
    single_timestep = {
        'visual': torch.randn(config['batch_size'], 1, 64)
    }
    outputs = gating(single_timestep)
    assert outputs['visual'].shape == single_timestep['visual'].shape
    
    # Test with invalid modality
    invalid_features = {
        'invalid_modality': torch.randn(
            config['batch_size'], 
            config['sequence_length'], 
            64
        )
    }
    outputs = gating(invalid_features)
    assert len(outputs) == 0


def test_modality_gating_different_sequence_lengths(config):
    """ Test with varying sequence lengths """
    gating = ModalityGating(
        feature_dims=config['feature_dims'],
        hidden_dim=config['hidden_dim']
    )
    
    # Create inputs with different sequence lengths
    varying_features = {
        'visual': torch.randn(config['batch_size'], 10, config['feature_dims']['visual']),
        'text': torch.randn(config['batch_size'], 15, config['feature_dims']['text'])
    }
    
    outputs = gating(varying_features)
    
    # Verify shapes are maintained
    assert outputs['visual'].shape == varying_features['visual'].shape
    assert outputs['text'].shape == varying_features['text'].shape