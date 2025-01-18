# test_multimodal_dynamic.py


""" Testing for dynamic fusion multimodal model definition """


import pytest
import torch
import torch.nn as nn

from omegaconf import DictConfig
from ocf_datapipes.batch import BatchKey, NWPBatchKey
from torch.optim import SGD

from pvnet.models.multimodal.multimodal_dynamic import Model
from pvnet.models.multimodal.linear_networks.output_networks import DynamicOutputNetwork


class MockNWPEncoder(nn.Module):
    """ Simplified mock encoder - explicit dimension handling """

    def __init__(self, in_channels=4, image_size_pixels=224):
        super().__init__()
        self.keywords = {"in_channels": in_channels}
        self.image_size_pixels = image_size_pixels
        self.hidden_dim = 256
        
        # Generate exact feature size needed
        self.features = nn.Parameter(torch.randn(self.hidden_dim))

    def forward(self, x):

        batch_size = x.size(0)        
        return self.features.unsqueeze(0).expand(batch_size, -1)


# Basic model as fixture - definition
@pytest.fixture
def basic_model():
    nwp_encoders_dict = {"mock_nwp": MockNWPEncoder()}
    nwp_forecast_minutes = DictConfig({"mock_nwp": 60})
    nwp_history_minutes = DictConfig({"mock_nwp": 60})
    
    model = Model(
        output_network=DynamicOutputNetwork,
        nwp_encoders_dict=nwp_encoders_dict,
        pv_encoder=None,
        wind_encoder=None,
        sensor_encoder=None,
        add_image_embedding_channel=False,
        include_gsp_yield_history=False,
        include_sun=False,
        include_time=False,
        embedding_dim=None,
        fusion_hidden_dim=256,
        num_fusion_heads=8,
        fusion_dropout=0.1,
        fusion_method="weighted_sum",
        forecast_minutes=30,
        history_minutes=60,
        nwp_forecast_minutes=nwp_forecast_minutes,
        nwp_history_minutes=nwp_history_minutes,
    )
    
    return model


def test_model_forward_pass(basic_model):
    """ Standard forward pass test """

    batch_size = 4
    sequence_length = basic_model.history_len
    height = width = 224
    channels = 4
    
    mock_nwp_data = torch.randn(batch_size, sequence_length, channels, height, width)
    batch = {
        BatchKey.nwp: {
            "mock_nwp": {
                NWPBatchKey.nwp: mock_nwp_data
            }
        }
    }
    
    with torch.no_grad():
        encoded_nwp = basic_model.nwp_encoders_dict["mock_nwp"](mock_nwp_data)
        print(f"Encoded NWP shape: {encoded_nwp.shape}")
        
        output, encoded_features = basic_model(batch)
    
    # Assert - check dimensions with forward pass
    assert output.shape == (batch_size, basic_model.num_output_features)
    assert isinstance(encoded_features, torch.Tensor)
    assert encoded_features.shape == (batch_size, basic_model.fusion_hidden_dim)


def test_model_init_minimal():
    """ Minimal initialisation of model test """

    nwp_encoders_dict = {"mock_nwp": MockNWPEncoder()}
    nwp_forecast_minutes = DictConfig({"mock_nwp": 60})
    nwp_history_minutes = DictConfig({"mock_nwp": 60})
    
    model = Model(
        output_network=DynamicOutputNetwork,
        nwp_encoders_dict=nwp_encoders_dict,
        pv_encoder=None,
        wind_encoder=None,
        sensor_encoder=None,
        add_image_embedding_channel=False,
        include_gsp_yield_history=False,
        include_sun=False,
        include_time=False,
        embedding_dim=None,
        fusion_hidden_dim=256,
        num_fusion_heads=8,
        fusion_dropout=0.1,
        fusion_method="weighted_sum",
        forecast_minutes=30,
        history_minutes=60,
        nwp_forecast_minutes=nwp_forecast_minutes,
        nwp_history_minutes=nwp_history_minutes,
    )
    
    assert isinstance(model, nn.Module)
    assert model.include_nwp
    assert not model.include_pv
    assert not model.include_wind
    assert not model.include_sensor
    assert not model.include_sun
    assert not model.include_time
    assert not model.include_gsp_yield_history
    
    assert isinstance(model.nwp_encoders_dict, dict)
    assert "mock_nwp" in model.nwp_encoders_dict
    
    assert isinstance(model.encoder, nn.Module)
    assert isinstance(model.output_network, nn.Module)


def test_model_quantile_regression(basic_model):
    """ Test model with quantile regression config """

    # Create model with quantile regression
    quantile_model = Model(
        output_network=DynamicOutputNetwork,
        output_quantiles=[0.1, 0.5, 0.9],
        nwp_encoders_dict={"mock_nwp": MockNWPEncoder()},
        nwp_forecast_minutes=DictConfig({"mock_nwp": 60}),
        nwp_history_minutes=DictConfig({"mock_nwp": 60}),
        pv_encoder=None,
        wind_encoder=None,
        sensor_encoder=None,
        add_image_embedding_channel=False,
        include_gsp_yield_history=False,
        include_sun=False,
        include_time=False,
        embedding_dim=None,
        fusion_hidden_dim=256,
        num_fusion_heads=8,
        fusion_dropout=0.1,
        fusion_method="weighted_sum",
        forecast_minutes=30,
        history_minutes=60
    )

    batch_size = 4
    sequence_length = quantile_model.history_len
    height = width = 224
    channels = 4

    mock_nwp_data = torch.randn(batch_size, sequence_length, channels, height, width)
    batch = {
        BatchKey.nwp: {
            "mock_nwp": {
                NWPBatchKey.nwp: mock_nwp_data
            }
        }
    }

    with torch.no_grad():
        output, encoded_features = quantile_model(batch)

    # Verify output shape and type are correct when using multiple quantiles
    assert quantile_model.use_quantile_regression
    assert len(quantile_model.output_quantiles) == 3
    assert output.shape == (batch_size, quantile_model.forecast_len, len(quantile_model.output_quantiles))
    assert torch.isfinite(output).all()

    # Random init variation check
    quantile_variances = output.std(dim=2)
    assert (quantile_variances > 0).any(), "Quantile predictions should show some variation"



def test_model_partial_inputs_and_error_handling(basic_model):
    """ Check error handling / robustness of model """

    batch_size = 4
    sequence_length = basic_model.history_len
    height = width = 224
    channels = 4

    # Minimal valid input
    minimal_batch = {
        BatchKey.nwp: {
            "mock_nwp": {
                NWPBatchKey.nwp: torch.randn(batch_size, sequence_length, channels, height, width)
            }
        }
    }

    with torch.no_grad():
        output, encoded_features = basic_model(minimal_batch)

    assert output.shape == (batch_size, basic_model.num_output_features)
    assert encoded_features.shape == (batch_size, basic_model.fusion_hidden_dim)
    assert torch.isfinite(output).all()

    # Missing NWP data
    empty_nwp_batch = {
        BatchKey.nwp: {}
    }

    with pytest.raises(Exception):
        with torch.no_grad():
            _ = basic_model(empty_nwp_batch)

    # None input for NWP
    none_nwp_batch = {
        BatchKey.nwp: {
            "mock_nwp": {
                NWPBatchKey.nwp: None
            }
        }
    }

    with pytest.raises(Exception):
        with torch.no_grad():
            _ = basic_model(none_nwp_batch)

    # Empty input dict
    empty_batch = {}

    with pytest.raises(Exception):
        with torch.no_grad():
            _ = basic_model(empty_batch)

    # Verify model can handle variations in input
    varied_sequence_batch = {
        BatchKey.nwp: {
            "mock_nwp": {
                NWPBatchKey.nwp: torch.randn(batch_size, max(1, sequence_length - 1), channels, height, width)
            }
        }
    }

    try:
        with torch.no_grad():
            result, _ = basic_model(varied_sequence_batch)
    except Exception as e:
        assert "input" in str(e).lower() or "shape" in str(e).lower()


def test_model_backward(basic_model):
    """ Test backward pass functionality - backprop verify """
    
    batch_size = 4
    sequence_length = basic_model.history_len
    height = width = 224
    channels = 4
    
    # Prepare input batch
    batch = {
        BatchKey.nwp: {
            "mock_nwp": {
                NWPBatchKey.nwp: torch.randn(batch_size, sequence_length, channels, height, width)
            }
        }
    }
    
    optimizer = SGD(basic_model.parameters(), lr=0.001)    
    output, _ = basic_model(batch)
    
    # Backward pass
    optimizer.zero_grad()
    output.sum().backward()
    
    # Check gradients are not None
    for name, param in basic_model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} is None"


def test_quantile_model_backward(basic_model):
    """ Test backward pass functionality - backprop verify - quantile regression """

    # Create model with quantile regression
    quantile_model = Model(
        output_network=DynamicOutputNetwork,
        output_quantiles=[0.1, 0.5, 0.9],
        nwp_encoders_dict={"mock_nwp": MockNWPEncoder()},
        nwp_forecast_minutes=DictConfig({"mock_nwp": 60}),
        nwp_history_minutes=DictConfig({"mock_nwp": 60}),
        pv_encoder=None,
        wind_encoder=None,
        sensor_encoder=None,
        add_image_embedding_channel=False,
        include_gsp_yield_history=False,
        include_sun=False,
        include_time=False,
        embedding_dim=None,
        fusion_hidden_dim=256,
        num_fusion_heads=8,
        fusion_dropout=0.1,
        fusion_method="weighted_sum",
        forecast_minutes=30,
        history_minutes=60
    )
    
    batch_size = 4
    sequence_length = quantile_model.history_len
    height = width = 224
    channels = 4
    
    # Prepare input batch
    batch = {
        BatchKey.nwp: {
            "mock_nwp": {
                NWPBatchKey.nwp: torch.randn(batch_size, sequence_length, channels, height, width)
            }
        }
    }
    
    optimizer = SGD(quantile_model.parameters(), lr=0.001)    
    output, _ = quantile_model(batch)
    
    # Backward pass
    optimizer.zero_grad()
    output.sum().backward()
    
    # Check quantile regression specific properties
    assert quantile_model.use_quantile_regression
    assert len(quantile_model.output_quantiles) == 3    
    assert output.shape == (batch_size, quantile_model.forecast_len, len(quantile_model.output_quantiles))
    
    # Check gradients are not None
    for name, param in quantile_model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} is None"
