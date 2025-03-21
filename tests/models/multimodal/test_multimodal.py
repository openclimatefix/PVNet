from torch.optim import SGD
import pytest
import torch

from pvnet.models.multimodal.multimodal import Model

def test_model_forward(multimodal_model, sample_batch):
    y = multimodal_model(sample_batch)

    # check output is the correct shape
    # batch size=2, forecast_len=15
    assert tuple(y.shape) == (2, 16), y.shape


def test_model_backward(multimodal_model, sample_batch):
    opt = SGD(multimodal_model.parameters(), lr=0.001)

    y = multimodal_model(sample_batch)

    # Backwards on sum drives sum to zero
    y.sum().backward()


def test_quantile_model_forward(multimodal_quantile_model, sample_batch):
    y_quantiles = multimodal_quantile_model(sample_batch)

    # check output is the correct shape
    # batch size=2, forecast_len=15, num_quantiles=3
    assert tuple(y_quantiles.shape) == (2, 16, 3), y_quantiles.shape


def test_quantile_model_backward(multimodal_quantile_model, sample_batch):
    opt = SGD(multimodal_quantile_model.parameters(), lr=0.001)

    y_quantiles = multimodal_quantile_model(sample_batch)

    # Backwards on sum drives sum to zero
    y_quantiles.sum().backward()


def test_weighted_quantile_model_forward(multimodal_quantile_model_ignore_minutes, sample_batch):
    y_quantiles = multimodal_quantile_model_ignore_minutes(sample_batch)

    # check output is the correct shape
    # batch size=2, forecast_len=8, num_quantiles=3
    assert tuple(y_quantiles.shape) == (2, 8, 3), y_quantiles.shape

    # Backwards on sum drives sum to zero
    y_quantiles.sum().backward()


def test_model_with_solar_position_config(multimodal_model_kwargs, sample_batch):
    """Test that the model automatically detects and uses solar positions based on config."""
    # Modify model kwargs - include solar position config
    model_kwargs = multimodal_model_kwargs.copy()
    model_kwargs["include_sun"] = True

    # Create model with the solar config
    model = Model(**model_kwargs)

    # Create test batch with only new keys
    batch_copy = sample_batch.copy()

    # Clear all existing solar keys
    for key in [
        "solar_azimuth",
        "solar_elevation",
    ]:
        if key in batch_copy:
            del batch_copy[key]

    batch_size = batch_copy["gsp"].shape[0]
    seq_len = model.forecast_len + model.history_len + 1
    batch_copy["solar_azimuth"] = torch.rand((batch_size, seq_len))
    batch_copy["solar_elevation"] = torch.rand((batch_size, seq_len))

    # The forward pass should work automatically based on config
    y = model(batch_copy)

    # Check output is the correct shape
    assert tuple(y.shape) == (2, 16), y.shape

    # Test backward pass
    y.sum().backward()
