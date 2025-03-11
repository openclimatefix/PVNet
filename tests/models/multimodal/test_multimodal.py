from torch.optim import SGD
import pytest


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


@pytest.mark.parametrize(
    "keys",
    [
        ["solar_azimuth", "solar_elevation"],
        ["gsp_solar_azimuth", "gsp_solar_elevation"],
    ],
)


def test_model_with_solar_position_keys(multimodal_model, sample_batch, keys):
    """Test that the model works with both new and legacy solar position keys."""
    azimuth_key, elevation_key = keys    
    batch_copy = sample_batch.copy()
    
    # Clear all solar keys and add just the ones we're testing
    for key in ["solar_azimuth", "solar_elevation", 
                "gsp_solar_azimuth", "gsp_solar_elevation"]:
        if key in batch_copy:
            del batch_copy[key]
    
    # Create solar position data if needed
    import torch
    batch_size = sample_batch["gsp"].shape[0]
    seq_len = multimodal_model.forecast_len + multimodal_model.history_len + 1
    batch_copy[azimuth_key] = torch.rand((batch_size, seq_len))
    batch_copy[elevation_key] = torch.rand((batch_size, seq_len))
    
    # Test forward and backward passes
    y = multimodal_model(batch_copy)    
    assert tuple(y.shape) == (2, 16), y.shape
    y.sum().backward()
