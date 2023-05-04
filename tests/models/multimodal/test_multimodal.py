from pvnet.models.multimodal.multimodal import Model
import pytest

@pytest.fixture()
def multimodal_model(multimodal_model_kwargs):
    model = Model(
        **multimodal_model_kwargs
    )
    return model


def test_model_forward(multimodal_model, sample_batch):
    
    y = multimodal_model(sample_batch)

    # check output is the correct shape
    # batch size=2, forecast_len=15
    assert tuple(y.shape)==(2, 16), y.shape