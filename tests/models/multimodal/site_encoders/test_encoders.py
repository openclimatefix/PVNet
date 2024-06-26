import torch
from ocf_datapipes.batch import BatchKey
from torch import nn

from pvnet.models.multimodal.site_encoders.encoders import (
    SimpleLearnedAggregator,
    SingleAttentionNetwork,
)

import pytest


def _test_model_forward(batch, model_class, kwargs):
    model = model_class(**kwargs)
    y = model(batch)
    assert tuple(y.shape) == (2, kwargs["out_features"]), y.shape


def _test_model_backward(batch, model_class, kwargs):
    model = model_class(**kwargs)
    y = model(batch)
    # Backwards on sum drives sum to zero
    y.sum().backward()


# Test model forward on all models
def test_simplelearnedaggregator_forward(sample_batch, site_encoder_model_kwargs):
    _test_model_forward(sample_batch, SimpleLearnedAggregator, site_encoder_model_kwargs)


def test_singleattentionnetwork_forward(sample_batch, site_encoder_model_kwargs):
    _test_model_forward(sample_batch, SingleAttentionNetwork, site_encoder_model_kwargs)


def test_singleattentionnetwork_forward_4d(sample_wind_batch, site_encoder_sensor_model_kwargs):
    _test_model_forward(sample_wind_batch, SingleAttentionNetwork, site_encoder_sensor_model_kwargs)


# Test model backward on all models
def test_simplelearnedaggregator_backward(sample_batch, site_encoder_model_kwargs):
    _test_model_backward(sample_batch, SimpleLearnedAggregator, site_encoder_model_kwargs)


def test_singleattentionnetwork_backward(sample_batch, site_encoder_model_kwargs):
    _test_model_backward(sample_batch, SingleAttentionNetwork, site_encoder_model_kwargs)
