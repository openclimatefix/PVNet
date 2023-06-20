from pvnet.models.multimodal.encoders.encodersRNN import (
    ConvLSTM,
    FlattenLSTM,
)
import pytest


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
def test_convlstm_forward(sample_satellite_batch, encoder_model_kwargs):
    # Skip if optional dependency not installed
    pytest.importorskip("metnet")
    _test_model_forward(sample_satellite_batch, ConvLSTM, encoder_model_kwargs)


def test_flattenlstm_forward(sample_satellite_batch, encoder_model_kwargs):
    _test_model_forward(sample_satellite_batch, FlattenLSTM, encoder_model_kwargs)


# Test model backward on all models
def test_convlstm_backward(sample_satellite_batch, encoder_model_kwargs):
    # Skip if optional dependency not installed
    pytest.importorskip("metnet")
    _test_model_backward(sample_satellite_batch, ConvLSTM, encoder_model_kwargs)


def test_flattenlstm_backward(sample_satellite_batch, encoder_model_kwargs):
    _test_model_backward(sample_satellite_batch, FlattenLSTM, encoder_model_kwargs)

