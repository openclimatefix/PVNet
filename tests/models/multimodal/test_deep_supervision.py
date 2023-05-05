from pvnet.models.multimodal.deep_supervision import Model
import pytest


@pytest.fixture()
def deepsupervision_model(multimodal_model_kwargs):
    model = Model(**multimodal_model_kwargs)
    return model


def test_model_forward(deepsupervision_model, sample_batch):

    y = deepsupervision_model(sample_batch)

    # check output is the correct shape
    # batch size=2, forecast_len=15
    assert tuple(y.shape) == (2, 16), y.shape
