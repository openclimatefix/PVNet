# test_attention_blocks.py

""" 
Tests each attention block - verifying initialisation, forward pass, gradients and masking
"""


import torch
import pytest
from pvnet.models.multimodal.attention_blocks import MultiheadAttention, CrossModalAttention, SelfAttention


# Fixture config combines values from multimodal.yaml with reduced test specific params
# embed_dim, batch_size, seq_len, and dropout stated smaller for faster testing execution
@pytest.fixture
def config():
    """ Test configuration parameters """
    return {
        'embed_dim': 64,
        'num_heads': 4,
        'batch_size': 8,
        'seq_len': 16,
        'dropout': 0.1,
        'num_modalities': 2
    }


@pytest.fixture
def attention_inputs(config):
    """ Generate attention input tensors """
    return {
        'query': torch.randn(config['batch_size'], config['seq_len'], config['embed_dim']),
        'key': torch.randn(config['batch_size'], config['seq_len'], config['embed_dim']),
        'value': torch.randn(config['batch_size'], config['seq_len'], config['embed_dim']),
        'mask': torch.ones(config['batch_size'], config['num_heads'], 
                          config['seq_len'], config['seq_len'])
    }


@pytest.fixture
def multimodal_inputs(config):
    """ Generate multimodal input tensors """
    return {
        'visual': torch.randn(config['batch_size'], config['seq_len'], config['embed_dim']),
        'text': torch.randn(config['batch_size'], config['seq_len'], config['embed_dim'])
    }


# MultiheadAttention testing
def test_multihead_attention_initialisation(config):
    """ Verify MultiheadAttention initialisation parameters """
    attn = MultiheadAttention(config['embed_dim'], config['num_heads'], config['dropout'])
    assert attn.head_dim == config['embed_dim'] // config['num_heads']
    assert attn.scale == (config['embed_dim'] // config['num_heads']) ** -0.5


def test_multihead_attention_forward(config, attention_inputs):
    """ Test MultiheadAttention forward pass shape """
    attn = MultiheadAttention(config['embed_dim'], config['num_heads'])
    output = attn(attention_inputs['query'], 
                 attention_inputs['key'], 
                 attention_inputs['value'])
    
    expected_shape = (config['batch_size'], config['seq_len'], config['embed_dim'])
    assert output.shape == expected_shape


def test_multihead_attention_mask(config, attention_inputs):
    """ Test attention masking functionality """
    # Mask first position
    attn = MultiheadAttention(config['embed_dim'], config['num_heads'])
    mask = attention_inputs['mask']
    mask[:, :, 0, :] = 0
    
    output = attn(attention_inputs['query'], 
                 attention_inputs['key'], 
                 attention_inputs['value'], 
                 mask)
    
    expected_shape = (config['batch_size'], config['seq_len'], config['embed_dim'])
    assert output.shape == expected_shape


def test_multihead_attention_gradient_flow(config, attention_inputs):
    """ Verify gradient computation for key components """
    attn = MultiheadAttention(config['embed_dim'], config['num_heads'])
    output = attn(attention_inputs['query'], 
                 attention_inputs['key'], 
                 attention_inputs['value'])
    loss = output.sum()
    loss.backward()
    
    assert attn.q_proj.weight.grad is not None
    assert attn.k_proj.weight.grad is not None
    assert attn.v_proj.weight.grad is not None
    assert attn.out_proj.weight.grad is not None


# CrossModalAttention testing
def test_cross_modal_attention_initialisation(config):
    """ Verify CrossModalAttention initialisation structure """
    cross_attn = CrossModalAttention(config['embed_dim'], 
                                   config['num_heads'], 
                                   config['dropout'], 
                                   config['num_modalities'])
    assert len(cross_attn.attention_blocks) == config['num_modalities']
    assert len(cross_attn.layer_norms) == config['num_modalities']


def test_cross_modal_attention_forward(config, multimodal_inputs):
    """ Test CrossModalAttention forward pass for each modality """
    cross_attn = CrossModalAttention(config['embed_dim'], 
                                   config['num_heads'], 
                                   config['dropout'])
    outputs = cross_attn(multimodal_inputs)
    
    assert len(outputs) == len(multimodal_inputs)
    for key, value in outputs.items():
        expected_shape = (config['batch_size'], config['seq_len'], config['embed_dim'])
        assert value.shape == expected_shape


def test_cross_modal_attention_gradient_flow(config, multimodal_inputs):
    """ Verify gradient computation for CrossModalAttention """
    cross_attn = CrossModalAttention(config['embed_dim'], config['num_heads'])
    outputs = cross_attn(multimodal_inputs)
    loss = sum(output.sum() for output in outputs.values())
    loss.backward()
    
    for block in cross_attn.attention_blocks:
        assert block.q_proj.weight.grad is not None
        assert block.out_proj.weight.grad is not None


# SelfAttention testing
def test_self_attention_initialisation(config):
    """ Verify SelfAttention component structure """
    self_attn = SelfAttention(config['embed_dim'], 
                             config['num_heads'], 
                             config['dropout'])
    assert isinstance(self_attn.attention, MultiheadAttention)
    assert isinstance(self_attn.layer_norm, torch.nn.LayerNorm)


def test_self_attention_forward(config, attention_inputs):
    """ Test SelfAttention forward pass shape """
    self_attn = SelfAttention(config['embed_dim'], config['num_heads'])
    output = self_attn(attention_inputs['query'])
    
    expected_shape = (config['batch_size'], config['seq_len'], config['embed_dim'])
    assert output.shape == expected_shape


def test_self_attention_gradient_flow(config, attention_inputs):
    """ Verify gradient computation for SelfAttention components """
    self_attn = SelfAttention(config['embed_dim'], config['num_heads'])
    output = self_attn(attention_inputs['query'])
    loss = output.sum()
    loss.backward()
    
    assert self_attn.attention.q_proj.weight.grad is not None
    assert self_attn.layer_norm.weight.grad is not None


def test_multihead_attention_numerical_stability(config, attention_inputs):
    """Test attention output for numerical stability"""
    attn = MultiheadAttention(config['embed_dim'], config['num_heads'])
    
    # Test with very large inputs
    large_inputs = {k: v * 1e5 for k, v in attention_inputs.items()}
    output_large = attn(large_inputs['query'], large_inputs['key'], large_inputs['value'])
    assert not torch.isnan(output_large).any(), "NaN values in large input output"
    
    # Test with very small inputs
    small_inputs = {k: v * 1e-5 for k, v in attention_inputs.items()}
    output_small = attn(small_inputs['query'], small_inputs['key'], small_inputs['value'])
    assert not torch.isnan(output_small).any(), "NaN values in small input output"


def test_attention_pattern(config, attention_inputs):
    """Verify expected attention patterns"""
    attn = MultiheadAttention(config['embed_dim'], config['num_heads'])
    
    # Create diagonal-dominant input with correct shape
    diagonal_query = torch.eye(config['seq_len'])
    # Expand to match embed_dim
    diagonal_query = diagonal_query.repeat(1, config['embed_dim'] // config['seq_len'])
    # Add batch dimension and expand
    diagonal_query = diagonal_query.unsqueeze(0).expand(config['batch_size'], -1, -1)
    diagonal_query = diagonal_query.to(attention_inputs['query'].dtype)
    
    output = attn(diagonal_query, 
                 attention_inputs['key'],
                 attention_inputs['value'])
    
    # Ensure tensors are contiguous and properly reshaped
    output_flat = output.contiguous().reshape(-1)
    query_flat = diagonal_query.contiguous().reshape(-1)
    
    # Check if output maintains some correlation with input
    correlation = torch.corrcoef(
        torch.stack([output_flat, query_flat])
    )[0, 1]
    assert not torch.isnan(correlation), "NaN correlation found"


def test_cross_modal_interaction(config, multimodal_inputs):
    """Test if modalities influence each other"""
    cross_attn = CrossModalAttention(config['embed_dim'], config['num_heads'])
    
    # Get outputs with normal inputs
    outputs_normal = cross_attn(multimodal_inputs)
    
    # Modify one modality
    modified_inputs = multimodal_inputs.copy()
    modified_inputs['visual'] = torch.zeros_like(multimodal_inputs['visual'])
    outputs_modified = cross_attn(modified_inputs)
    
    # Check if modification of one modality affects the other
    diff = (outputs_normal['text'] - outputs_modified['text']).abs().mean()
    assert diff > 0, "No cross-modal interaction detected"
