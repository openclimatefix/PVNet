from pvnet.models.multimodal.deep_supervision import Model
from torch.optim import SGD
import pytest


@pytest.fixture()
def deepsupervision_model(multimodal_model_kwargs):
    model = Model(**multimodal_model_kwargs)
    return model

@pytest.mark.skip(reason="This model is no longer maintained")
def test_model_forward(deepsupervision_model, sample_batch):
    y = deepsupervision_model(sample_batch)

    # check output is the correct shape
    # batch size=2, forecast_len=15
    assert tuple(y.shape) == (2, 16), y.shape

@pytest.mark.skip(reason="This model is no longer maintained")
def test_model_backwards(deepsupervision_model, sample_batch):
    opt = SGD(deepsupervision_model.parameters(), lr=0.001)

    y = deepsupervision_model(sample_batch)

    # Backwards on sum drives sum to zero
    y.sum().backward()
