from torch.optim import SGD
import math
import pytest


def test_model_forward(multimodal_model, sample_batch):
    y = multimodal_model(sample_batch)

    # check output is the correct shape
    # batch size=2, forecast_len=15
    assert tuple(y.shape) == (2, 16), y.shape

def test_model_forward_site_history(multimodal_model_site_history, sample_site_batch):

    y = multimodal_model_site_history(sample_site_batch)

    # check output is the correct shape
    # batch size=2, forecast_len=15
    assert tuple(y.shape) == (2, 16), y.shape


def test_model_backward(multimodal_model, sample_batch):
    opt = SGD(multimodal_model.parameters(), lr=0.001)

    y = multimodal_model(sample_batch)

    # Backwards on sum drives sum to zero
    y.sum().backward()


def test_quantile_model_forward(multimodal_quantile_model, sample_batch):
    y_quantiles = multimodal_quantile_model(sample_batch)

    # check output is the correct shape
    # batch size=2, forecast_len=15, num_quantiles=3
    assert tuple(y_quantiles.shape) == (2, 16, 3), y_quantiles.shape


def test_quantile_model_backward(multimodal_quantile_model, sample_batch):
    opt = SGD(multimodal_quantile_model.parameters(), lr=0.001)

    y_quantiles = multimodal_quantile_model(sample_batch)

    # Backwards on sum drives sum to zero
    y_quantiles.sum().backward()


def test_weighted_quantile_model_forward(multimodal_quantile_model_ignore_minutes, sample_batch):
    y_quantiles = multimodal_quantile_model_ignore_minutes(sample_batch)

    # check output is the correct shape
    # batch size=2, forecast_len=8, num_quantiles=3
    assert tuple(y_quantiles.shape) == (2, 8, 3), y_quantiles.shape

    # Backwards on sum drives sum to zero
    y_quantiles.sum().backward()


# FOLLOWING TEST TO NOT BE INCLUDED
# PURELY FOR EXPERIMENTAL TESTING / SANITY CHECK

@pytest.mark.parametrize(
    "adaptive_model_setup",
    [
        {"wd_schedule": "cosine"},
        {"wd_schedule": "linear"},
        {"wd_schedule": "exponential"},
    ],
    indirect=True,
)
def test_adaptive_optimizer_and_wd_schedule(adaptive_model_setup):
    model = adaptive_model_setup["model"]
    optimizer_instance = adaptive_model_setup["optimizer_instance"]
    total_epochs = adaptive_model_setup["total_epochs"]
    wd_initial = adaptive_model_setup["wd_initial"]
    wd_final = adaptive_model_setup["wd_final"]
    wd_schedule = adaptive_model_setup["wd_schedule"]
    initial_lr = adaptive_model_setup["initial_lr"]
    lr_scaling_factor = adaptive_model_setup["lr_scaling_factor"]

    found_embedding_group = False
    found_depth1_group = False
    found_depth2_group = False

    for group in optimizer_instance.param_groups:
        current_lr = group['lr']
        current_wd = group['weight_decay']

        if current_wd == 0.0:
            found_embedding_group = True
            assert abs(current_lr - initial_lr) < 1e-6

        else:
            if abs(current_lr - (initial_lr * (lr_scaling_factor ** 1))) < 1e-6:
                found_depth1_group = True
                assert abs(current_wd - wd_initial) < 1e-6
            elif abs(current_lr - (initial_lr * (lr_scaling_factor ** 2))) < 1e-6:
                found_depth2_group = True
                assert abs(current_wd - wd_initial) < 1e-6
            else:
                raise AssertionError(f"Unexpected parameter group found: LR={current_lr}, WD={current_wd}")

    assert found_embedding_group
    assert found_depth1_group
    assert found_depth2_group

    for epoch in range(total_epochs):
        if wd_schedule == "cosine":
            expected_wd = wd_final + 0.5 * (wd_initial - wd_final) * (1 + math.cos(math.pi * (epoch / total_epochs)))
        elif wd_schedule == "linear":
            expected_wd = wd_initial * (1 - (epoch / total_epochs)) + wd_final * (epoch / total_epochs)
        elif wd_schedule == "exponential":
            if wd_initial == 0:
                 expected_wd = wd_final
            else:
                 expected_wd = wd_initial * ((wd_final / wd_initial) ** (epoch / total_epochs))
        else:
            raise ValueError(f"Unknown weight decay schedule type: {wd_schedule}")

        model.trainer.current_epoch = epoch
        model.on_train_epoch_start()

        for group in optimizer_instance.param_groups:
            if group.get('weight_decay') != 0.0:
                assert abs(group['weight_decay'] - expected_wd) < 1e-6
            else:
                assert group['weight_decay'] == 0.0
