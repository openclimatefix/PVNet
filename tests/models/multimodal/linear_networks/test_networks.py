from pvnet.models.multimodal.linear_networks.networks import (
    DefaultFCNet,
    ResFCNet,
    ResFCNet2,
    SNN,
    TabNet,
)
import pytest
import torch
from collections import OrderedDict


@pytest.fixture()
def simple_linear_batch():
    return torch.rand(2, 100)


@pytest.fixture()
def multimodal_linear_batch():
    return OrderedDict(nwp=torch.rand(2, 50), sat=torch.rand(2, 40), sun=torch.rand(2, 10))


@pytest.fixture()
def multiple_batch_types(simple_linear_batch, multimodal_linear_batch):
    return [simple_linear_batch, multimodal_linear_batch]


@pytest.fixture()
def fc_batch_batch():
    return torch.rand(2, 100)


@pytest.fixture()
def linear_network_kwargs():
    kwargs = dict(in_features=100, out_features=10)
    return kwargs


def _test_model_forward(batches, model_class, model_kwargs):
    for batch in batches:
        model = model_class(**model_kwargs)
        y = model(batch)
        assert tuple(y.shape) == (2, model_kwargs["out_features"]), y.shape


def _test_model_backward(batch, model_class, model_kwargs):
    model = model_class(**model_kwargs)
    y = model(batch)
    # Backwards on sum drives sum to zero
    y.sum().backward()


# Test model forward on all models
def test_defaultfcnet_forward(multiple_batch_types, linear_network_kwargs):
    _test_model_forward(multiple_batch_types, DefaultFCNet, linear_network_kwargs)


def test_resfcnet_forward(multiple_batch_types, linear_network_kwargs):
    _test_model_forward(multiple_batch_types, ResFCNet, linear_network_kwargs)


def test_resfcnet2_forward(multiple_batch_types, linear_network_kwargs):
    _test_model_forward(multiple_batch_types, ResFCNet2, linear_network_kwargs)


def test_snn_forward(multiple_batch_types, linear_network_kwargs):
    _test_model_forward(multiple_batch_types, SNN, linear_network_kwargs)


def test_tabnet_forward(multiple_batch_types, linear_network_kwargs):
    # Skip if optional dependency not installed
    pytest.importorskip("pytorch_tabnet") 
    _test_model_forward(multiple_batch_types, TabNet, linear_network_kwargs)


# Test model backward on all models
def test_defaultfcnet_backward(simple_linear_batch, linear_network_kwargs):
    _test_model_backward(simple_linear_batch, DefaultFCNet, linear_network_kwargs)


def test_resfcnet_backward(simple_linear_batch, linear_network_kwargs):
    _test_model_backward(simple_linear_batch, ResFCNet, linear_network_kwargs)


def test_resfcnet2_backward(simple_linear_batch, linear_network_kwargs):
    _test_model_backward(simple_linear_batch, ResFCNet2, linear_network_kwargs)


def test_snn_backward(simple_linear_batch, linear_network_kwargs):
    _test_model_backward(simple_linear_batch, SNN, linear_network_kwargs)


def test_tabnet_backward(simple_linear_batch, linear_network_kwargs):
    # Skip if optional dependency not installed
    pytest.importorskip("pytorch_tabnet") 
    _test_model_backward(simple_linear_batch, TabNet, linear_network_kwargs)
