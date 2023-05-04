from pvnet.models.multimodal.nwp_weighting import Model
import pytest

@pytest.fixture()
def nwp_weighting_model(model_minutes_kwargs):
    model = Model(
        **model_minutes_kwargs,
        nwp_image_size_pixels=24,
        dwsrf_channel=1,
    )
    return model


def test_model_forward(nwp_weighting_model, sample_batch):
    
    y = nwp_weighting_model(sample_batch)

    # check output is the correct shape
    # batch size=2, forecast_len=15
    assert tuple(y.shape)==(2, 16), y.shape
