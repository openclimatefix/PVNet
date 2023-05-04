from pvnet.models.baseline.single_value import Model
import pytest


@pytest.fixture()
def single_value_model(model_minutes_kwargs):
    model = Model(**model_minutes_kwargs)
    return model


def test_model_forward(last_value_model, sample_batch):

    y = single_value_model(sample_batch)

    # check output is the correct shape
    # batch size=2, forecast_len=15
    assert tuple(y.shape)==(2, 16), y.shape
