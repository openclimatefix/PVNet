import pytest
import torch

from pvnet.models.utils import WeightedLosses


def test_weight_losses_weights():
    """Test weighted loss"""
    forecast_length = 2
    w = WeightedLosses(forecast_length=forecast_length)

    assert w.weights.cpu().numpy()[0] == pytest.approx(4 / 3)
    assert w.weights.cpu().numpy()[1] == pytest.approx(2 / 3)


def test_mae_exp():
    """Test MAE exp with weighted loss"""
    forecast_length = 2
    w = WeightedLosses(forecast_length=forecast_length)

    output = torch.Tensor([[1, 3], [1, 3]])
    target = torch.Tensor([[1, 5], [1, 9]])

    loss = w.get_mae_exp(output=output, target=target)

    # 0.5((1-1)*2/3 + (5-3)*1/3) + 0.5((1-1)*2/3 + (9-3)*1/3) = 1/3 + 3/3
    assert loss == pytest.approx(4 / 3)


def test_mse_exp():
    """Test MSE exp with weighted loss"""
    forecast_length = 2
    w = WeightedLosses(forecast_length=forecast_length)

    output = torch.Tensor([[1, 3], [1, 3]])
    target = torch.Tensor([[1, 5], [1, 9]])

    loss = w.get_mse_exp(output=output, target=target)

    # 0.5((1-1)^2*2/3 + (5-3)^2*1/3) + 0.5((1-1)^2*2/3 + (9-3)^2*1/3) = 2/3 + 18/3
    assert loss == pytest.approx(20 / 3)


def test_mae_exp_rand():
    """Test MAE exp with weighted loss  with random tensors"""
    forecast_length = 6
    batch_size = 32

    w = WeightedLosses(forecast_length=6)

    output = torch.randn(batch_size, forecast_length)
    target = torch.randn(batch_size, forecast_length)

    loss = w.get_mae_exp(output=output, target=target)
    assert loss > 0


def test_mse_exp_rand():
    """Test MSE exp with weighted loss  with random tensors"""
    forecast_length = 6
    batch_size = 32

    w = WeightedLosses(forecast_length=6)

    output = torch.randn(batch_size, forecast_length)
    target = torch.randn(batch_size, forecast_length)

    loss = w.get_mse_exp(output=output, target=target)
    assert loss > 0
