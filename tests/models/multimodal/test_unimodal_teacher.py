import os
import glob
import tempfile
import yaml

import hydra
import pytest

from torch.optim import SGD
import torch

import pvnet
from pvnet.models.multimodal.unimodal_teacher import Model


@pytest.fixture
def teacher_dir(multimodal_model, raw_multimodal_model_kwargs):
    raw_multimodal_model_kwargs["_target_"] = "pvnet.models.multimodal.multimodal.Model"

    with tempfile.TemporaryDirectory() as tmpdirname:
        # Save teachers for these modes
        for mode in ["sat", "nwp_ukv"]:
            mode_dir = f"{tmpdirname}/{mode}"
            os.mkdir(mode_dir)

            # Checkpoint paths would be like: epoch={X}-step={N}.ckpt or last.ckpt
            path = os.path.join(mode_dir, "epoch=2-step=35002.ckpt")
            path = f"{mode_dir}/epoch=2-step=35002.ckpt"

            # Save out themodel config file
            with open(os.path.join(mode_dir, "model_config.yaml"), "w") as outfile:
                yaml.dump(raw_multimodal_model_kwargs, outfile)

            # Save the weights
            torch.save({"model_state_dict": multimodal_model.state_dict()}, path)

        yield tempfile


@pytest.fixture
def unimodal_model_kwargs(teacher_dir, model_minutes_kwargs):
    # Configure the fusion network
    kwargs = dict(
        output_network=dict(
            _target_=pvnet.models.multimodal.linear_networks.networks.ResFCNet2,
            _partial_=True,
            fc_hidden_features=128,
            n_res_blocks=6,
            res_block_layers=2,
            dropout_frac=0.0,
        ),
        cold_start=True,
    )

    # Get the teacher model save directories
    mode_dirs = glob.glob(f"{teacher_dir}/*")
    mode_teacher_dict = dict()
    for mode_dir in mode_dirs:
        mode_name = mode_dir.split("/")[-1].replace("nwp_", "nwp/")
        mode_teacher_dict[mode_name] = mode_dir
    kwargs["mode_teacher_dict"] = mode_teacher_dict

    # Add the forecast and history minutes to be compatible with the sample batch
    kwargs.update(model_minutes_kwargs)

    yield hydra.utils.instantiate(kwargs)


@pytest.fixture
def unimodal_teacher_model(unimodal_model_kwargs):
    return Model(**unimodal_model_kwargs)


def test_model_init(unimodal_model_kwargs):
    Model(**unimodal_model_kwargs)


def test_model_forward(unimodal_teacher_model, sample_batch):
    # assert False
    y, _ = unimodal_teacher_model(sample_batch, return_modes=True)

    # check output is the correct shape
    # batch size=2, forecast_len=15
    assert tuple(y.shape) == (2, 16), y.shape


def test_model_backward(unimodal_teacher_model, sample_batch):
    opt = SGD(unimodal_teacher_model.parameters(), lr=0.001)

    y = unimodal_teacher_model(sample_batch)

    # Backwards on sum drives sum to zero
    y.sum().backward()


def test_model_conversion(unimodal_model_kwargs, sample_batch):
    # Create the unimodal model
    um_model = Model(**unimodal_model_kwargs)
    # Convert to the equivalent multimodel model
    mm_model, _ = um_model.convert_to_multimodal_model(unimodal_model_kwargs)

    # If the model has been successfully converted the predictions should be identical
    y_um = um_model(sample_batch, return_modes=False)
    y_mm = mm_model(sample_batch)

    assert (y_um == y_mm).all()


def test_unimodal_model_with_solar_position(unimodal_model_kwargs, sample_batch):
    """Test that the unimodal model works with solar position data."""
    # Modify model kwargs - ensure sun is included
    model_kwargs = unimodal_model_kwargs.copy()
    model_kwargs["include_sun"] = True
    
    # Create model with sun enabled
    model = Model(**model_kwargs)

    # Create test batch with only new keys
    batch_copy = sample_batch.copy()

    # Clear all existing solar keys
    for key in [
        "solar_azimuth",
        "solar_elevation",
        "gsp_solar_azimuth",
        "gsp_solar_elevation",
    ]:
        if key in batch_copy:
            del batch_copy[key]

    # Create solar position data with new keys
    batch_size = batch_copy["gsp"].shape[0]
    seq_len = model.forecast_len + model.history_len + 1
    batch_copy["solar_azimuth"] = torch.rand((batch_size, seq_len))
    batch_copy["solar_elevation"] = torch.rand((batch_size, seq_len))

    # Test forward pass
    y = model(batch_copy)
    assert tuple(y.shape) == (2, 16), y.shape

    # Test backward pass
    y.sum().backward()
