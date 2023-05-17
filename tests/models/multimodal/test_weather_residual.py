from pvnet.models.multimodal.weather_residual import Model
from torch.optim import SGD
import pytest


@pytest.fixture()
def weather_residual_model(multimodal_model_kwargs):
    model = Model(**multimodal_model_kwargs)
    return model


def test_model_forward(weather_residual_model, sample_batch):

    y = weather_residual_model(sample_batch)

    # check output is the correct shape
    # batch size=2, forecast_len=15
    assert tuple(y.shape) == (2, 16), y.shape
    
    
def test_model_backwards(weather_residual_model, sample_batch):
    opt = SGD(weather_residual_model.parameters(), lr=0.001)

    y = weather_residual_model(sample_batch)
    
    # Backwards on sum drives sum to zero
    y.sum().backward()

