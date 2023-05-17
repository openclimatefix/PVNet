import pytest
from ocf_datapipes.utils.consts import BatchKey
import torch

from pvnet.data.datamodule import DataModule
import pvnet


@pytest.fixture()
def sample_datamodule():
    dm = DataModule(
        configuration=None,
        batch_size=2,
        num_workers=0,
        prefetch_factor=2,
        train_period=[None, None],
        val_period=[None, None],
        test_period=[None, None],
        block_nwp_and_sat=False,
        batch_dir="tests/data/sample_batches",
    )
    return dm


@pytest.fixture()
def sample_batch(sample_datamodule):
    batch = next(iter(sample_datamodule.train_dataloader()))
    return batch


@pytest.fixture()
def sample_satellite_batch(sample_batch):
    sat_image = sample_batch[BatchKey.satellite_actual]
    return torch.swapaxes(sat_image, 1, 2)


@pytest.fixture()
def model_minutes_kwargs():
    kwargs = dict(
        forecast_minutes=480,
        history_minutes=120,
    )
    return kwargs


@pytest.fixture()
def encoder_model_kwargs():
    # Used to test encoder model on satellite data
    kwargs = dict(
        sequence_length=90 // 5 - 2,
        image_size_pixels=24,
        in_channels=11,
        out_features=128,
    )
    return kwargs


@pytest.fixture()
def multimodal_model_kwargs(model_minutes_kwargs):
    kwargs = dict(
        image_encoder=pvnet.models.multimodal.encoders.encoders3d.DefaultPVNet,
        encoder_out_features=128,
        encoder_kwargs=dict(
            number_of_conv3d_layers=6,
            conv3d_channels=32,
        ),
        include_sat=True,
        include_nwp=True,
        add_image_embedding_channel=True,
        sat_image_size_pixels=24,
        nwp_image_size_pixels=24,
        number_sat_channels=11,
        number_nwp_channels=2,
        output_network=pvnet.models.multimodal.linear_networks.networks.ResFCNet2,
        output_network_kwargs=dict(
            fc_hidden_features=128,
            n_res_blocks=6,
            res_block_layers=2,
            dropout_frac=0.0,
        ),
        embedding_dim=16,
        include_sun=True,
        include_gsp_yield_history=True,
        sat_history_minutes=90,
        nwp_history_minutes=120,
        nwp_forecast_minutes=480,
    )
    kwargs.update(model_minutes_kwargs)
    return kwargs
