from pvnet.models.multimodal.multimodal import Model
from torch.optim import SGD
import pytest


@pytest.fixture()
def multimodal_model(multimodal_model_kwargs):
    model = Model(**multimodal_model_kwargs)
    return model


@pytest.fixture()
def multimodal_quantile_model(multimodal_model_kwargs):
    model = Model(output_quantiles=[0.1, 0.5, 0.9], **multimodal_model_kwargs)
    return model


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
