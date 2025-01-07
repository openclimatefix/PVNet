# test_output_networks.py

""" 
Tests DynamicOutputNetwork and QuantileOutputNetwork

Verifying initialisation, forward pass, residuals, and activation functionalities
"""


import torch
import pytest
from pvnet.models.multimodal.linear_networks.output_networks import DynamicOutputNetwork, QuantileOutputNetwork


# Fixture config
@pytest.fixture
def config():
    """Test configuration parameters."""
    return {
        'in_features': 128,
        'out_features': 10,
        'hidden_dims': [256, 128],
        'dropout': 0.1,
        'use_layer_norm': True,
        'use_residual': True
    }


@pytest.fixture
def inputs():
    """Fixture for input tensor."""
    return torch.ones(16, 128)


# DynamicOutputNetwork testing
def test_dynamic_output_network_initialization(config):
    """ Verify initialisation """
    network = DynamicOutputNetwork(
        in_features=config['in_features'],
        out_features=config['out_features'],
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout'],
        use_layer_norm=config['use_layer_norm'],
        use_residual=config['use_residual'],
        output_activation="softmax"
    )
    assert len(network.layers) == len(config['hidden_dims'])
    assert isinstance(network.output_layer, torch.nn.Linear)
    assert isinstance(network.output_activation, torch.nn.Softmax)


def test_dynamic_output_network_forward(config, inputs):
    """ Test forward pass """
    network = DynamicOutputNetwork(
        in_features=config['in_features'],
        out_features=config['out_features'],
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout'],
        use_layer_norm=config['use_layer_norm'],
        use_residual=False
    )
    output = network(inputs)
    assert output.shape == (inputs.shape[0], config['out_features'])


def test_dynamic_output_network_residual(config, inputs):
    """ Verify residual connection """
    network = DynamicOutputNetwork(
        in_features=config['in_features'],
        out_features=config['in_features'],
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout'],
        use_layer_norm=config['use_layer_norm'],
        use_residual=config['use_residual']
    )
    output = network(inputs)
    assert output.shape == inputs.shape


def test_dynamic_output_network_quantile_reshape(config, inputs):
    """ Check quantile output reshape """
    network = DynamicOutputNetwork(
        in_features=config['in_features'],
        out_features=config['out_features'],
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout'],
        quantile_output=True,
        num_forecast_steps=24
    )
    output = network(inputs)
    assert output.shape == (inputs.shape[0], 24, config['out_features'])


def test_output_activation_softmax(config, inputs):
    """ Verify softmax activation """
    network = DynamicOutputNetwork(
        in_features=config['in_features'],
        out_features=config['out_features'],
        hidden_dims=[256],
        output_activation="softmax"
    )
    output = network(inputs)
    assert torch.allclose(output.sum(dim=-1), torch.ones(inputs.shape[0]), atol=1e-5)


def test_output_activation_sigmoid(config, inputs):
    """ Verify sigmoid activation """
    network = DynamicOutputNetwork(
        in_features=config['in_features'],
        out_features=config['out_features'],
        hidden_dims=[256],
        output_activation="sigmoid"
    )
    output = network(inputs)
    assert torch.all((output >= 0) & (output <= 1))


# QuantileOutputNetwork testing
def test_quantile_output_network_forward(config, inputs):
    """ Test forward pass """
    network = QuantileOutputNetwork(
        in_features=config['in_features'],
        num_quantiles=3,
        num_forecast_steps=24,
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout']
    )
    output = network(inputs)
    assert output.shape == (inputs.shape[0], 24, 3)