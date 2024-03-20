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


def test_weighted_quantile_model_forward(multimodal_weighted_quantile_model, sample_batch):
    y_quantiles = multimodal_weighted_quantile_model(sample_batch)

    # check output is the correct shape
    # batch size=2, forecast_len=15, num_quantiles=3
    assert tuple(y_quantiles.shape) == (2, 16, 3), y_quantiles.shape


def test_weighted_quantile_model_backward(multimodal_weighted_quantile_model, sample_batch):
    opt = SGD(multimodal_weighted_quantile_model.parameters(), lr=0.001)

    y_quantiles = multimodal_weighted_quantile_model(sample_batch)

    # Backwards on sum drives sum to zero
    y_quantiles.sum().backward()
