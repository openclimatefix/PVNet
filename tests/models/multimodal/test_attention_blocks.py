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


# Error handling testing
def test_invalid_multihead_config():
    """ Test error handling for invalid configuration """
    with pytest.raises(ValueError, match="embed_dim not divisible by num_heads"):
        MultiheadAttention(embed_dim=65, num_heads=4)
