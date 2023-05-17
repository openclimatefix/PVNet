from pvnet.models.multimodal.encoders.encoders2d import (
    NaiveEfficientNet,
    NaiveResNet,
    ConvNeXt,
    CNBlockConfig,
)
import pytest

@pytest.fixture()
def convnext_model_kwargs(encoder_model_kwargs):
    model_kwargs = {k:v for k,v in encoder_model_kwargs.items()}
    model_kwargs["block_setting"] = [
            CNBlockConfig(96, 192, 3),
            CNBlockConfig(192, 384, 3),
            CNBlockConfig(384, 768, 9),
            CNBlockConfig(768, None, 3),
    ]
    return model_kwargs

def _test_model_forward(batch, model_class, model_kwargs):
    model = model_class(**model_kwargs)
    y = model(batch)
    assert tuple(y.shape) == (2, model_kwargs["out_features"]), y.shape

def _test_model_backward(batch, model_class, model_kwargs):
    model = model_class(**model_kwargs)
    y = model(batch)
    # Backwards on sum drives sum to zero
    y.sum().backward()


# Test model forward on all models
def test_naiveefficientnet_forward(sample_satellite_batch, encoder_model_kwargs):
    _test_model_forward(sample_satellite_batch, NaiveEfficientNet, encoder_model_kwargs)

def test_naiveresnet_forward(sample_satellite_batch, encoder_model_kwargs):
    _test_model_forward(sample_satellite_batch, NaiveResNet, encoder_model_kwargs)

def test_convnext_forward(sample_satellite_batch, convnext_model_kwargs):
    _test_model_forward(sample_satellite_batch, ConvNeXt, convnext_model_kwargs)

# Test model backward on all models
def test_naiveefficientnet_backward(sample_satellite_batch, encoder_model_kwargs):
    _test_model_backward(sample_satellite_batch, NaiveEfficientNet, encoder_model_kwargs)

def test_naiveresnet_backward(sample_satellite_batch, encoder_model_kwargs):
    _test_model_backward(sample_satellite_batch, NaiveResNet, encoder_model_kwargs)

def test_convnext_backward(sample_satellite_batch, convnext_model_kwargs):
    _test_model_backward(sample_satellite_batch, ConvNeXt, convnext_model_kwargs)

