# test_fusion_blocks.py

""" 
Tests fusion block components - verifying initialisation, forward pass, gradient flow, and fusion methods
"""


import torch
import pytest
from pvnet.models.multimodal.fusion_blocks import DynamicFusionModule, ModalityGating


# Fixture config combines values from multimodal.yaml with reduced test specific params
# batch_size, seq_len, and dropout stated smaller for faster testing execution
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
        'seq_len': 16,
        'dropout': 0.1
    }


@pytest.fixture
def multimodal_features(config):
    """ Generate multimodal input features """
    batch_size = config['batch_size']
    return {
        'visual': torch.randn(batch_size, config['feature_dims']['visual']),
        'text': torch.randn(batch_size, config['feature_dims']['text']),
        'audio': torch.randn(batch_size, config['feature_dims']['audio'])
    }


@pytest.fixture
def attention_mask(config):
    """ Generate attention mask for fusion weights """
    batch_size = config['batch_size']
    num_modalities = len(config['feature_dims'])
    return torch.ones(batch_size, num_modalities, dtype=torch.bool)


# DynamicFusionModule testing
def test_dynamic_fusion_initialization(config):
    """ Verify initialisation """
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


def test_dynamic_fusion_forward_weighted_sum(config, multimodal_features):
    """ Forward pass with weighted sum fusion """
    fusion = DynamicFusionModule(
        feature_dims=config['feature_dims'],
        hidden_dim=config['hidden_dim'],
        fusion_method="weighted_sum"
    )
    output = fusion(multimodal_features)
    
    # Assert output shape
    assert output.shape == (config['batch_size'], config['hidden_dim'])


def test_dynamic_fusion_forward_concat(config, multimodal_features):
    """ Forward pass with concat fusion """
    fusion = DynamicFusionModule(
        feature_dims=config['feature_dims'],
        hidden_dim=config['hidden_dim'],
        fusion_method="concat"
    )
    output = fusion(multimodal_features)
    
    # Assert output shape
    assert output.shape == (config['batch_size'], config['hidden_dim'])


def test_dynamic_fusion_masked_attention(config, multimodal_features, attention_mask):
    """ Test attention masking """
    fusion = DynamicFusionModule(
        feature_dims=config['feature_dims'],
        hidden_dim=config['hidden_dim']
    )
    
    attention_mask[0, 0] = False
    output = fusion(multimodal_features, attention_mask)
    
    # Assert output shape / values
    assert output.shape == (config['batch_size'], config['hidden_dim'])
    assert not torch.isnan(output).any()


def test_dynamic_fusion_gradient_flow(config, multimodal_features):
    """ Verify gradient computation """
    fusion = DynamicFusionModule(
        feature_dims=config['feature_dims'],
        hidden_dim=config['hidden_dim']
    )
    output = fusion(multimodal_features)
    loss = output.sum()
    loss.backward()
    
    # Assert gradients for key components
    for proj in fusion.projections.values():
        assert proj[0].weight.grad is not None
    assert fusion.cross_attention.q_proj.weight.grad is not None
    assert fusion.weight_network[0].weight.grad is not None


def test_dynamic_fusion_residual(config, multimodal_features):
    """ Test residual connection """
    fusion = DynamicFusionModule(
        feature_dims=config['feature_dims'],
        hidden_dim=config['hidden_dim'],
        use_residual=True
    )
    output = fusion(multimodal_features)
    
    assert output.shape == (config['batch_size'], config['hidden_dim'])
    assert hasattr(fusion, 'layer_norm')


def test_dynamic_fusion_invalid_method():
    """ Error handling for invalid parameters """
    with pytest.raises(ValueError, match="Invalid fusion method"):
        DynamicFusionModule(
            feature_dims={'visual': 64},
            fusion_method="invalid_method"
        )


def test_dynamic_fusion_empty_features(config):
    """ Handling of empty feature input """
    fusion = DynamicFusionModule(
        feature_dims=config['feature_dims'],
        hidden_dim=config['hidden_dim']
    )
    with pytest.raises(ValueError, match="Invalid features"):
        fusion({})


# ModalityGating testing
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


def test_modality_gating_forward(config, multimodal_features):
    """ Test forward pass """
    gating = ModalityGating(
        feature_dims=config['feature_dims'],
        hidden_dim=config['hidden_dim']
    )
    outputs = gating(multimodal_features)
    
    # Verify / assert output shapes and gating properties / ranges
    assert len(outputs) == len(multimodal_features)
    for modality, output in outputs.items():
        assert output.shape == multimodal_features[modality].shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


def test_modality_gating_gradient_flow(config, multimodal_features):
    """ Verify gradient computation """
    gating = ModalityGating(
        feature_dims=config['feature_dims'],
        hidden_dim=config['hidden_dim']
    )
    outputs = gating(multimodal_features)
    loss = sum(output.sum() for output in outputs.values())
    loss.backward()
    
    # Assert gradients
    for network in gating.gate_networks.values():
        assert network[0].weight.grad is not None


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


def test_modality_gating_edge_cases():
    """ Test edge cases """
    gating = ModalityGating(feature_dims={'visual': 64})
    
    # Empty input - return empty dict
    empty_outputs = gating({})
    assert isinstance(empty_outputs, dict)
    assert len(empty_outputs) == 0
    
    # Invalid modality - ignored
    invalid_outputs = gating({'invalid_modality': torch.randn(8, 64)})
    assert len(invalid_outputs) == 0
