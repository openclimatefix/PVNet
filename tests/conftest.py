import os
import glob
import tempfile

import pytest
import pandas as pd
import numpy as np
import xarray as xr
import torch
import hydra

from datetime import timedelta

import pvnet
from pvnet.data import DataModule, SiteDataModule

import pvnet.models.multimodal.encoders.encoders3d
import pvnet.models.multimodal.linear_networks.networks
import pvnet.models.multimodal.site_encoders.encoders
from pvnet.models.multimodal.multimodal import Model


xr.set_options(keep_attrs=True)


def time_before_present(dt: timedelta):
    return pd.Timestamp.now(tz=None) - dt


@pytest.fixture
def nwp_data():
    # Load dataset which only contains coordinates, but no data
    ds = xr.open_zarr(
        f"{os.path.dirname(os.path.abspath(__file__))}/test_data/sample_data/nwp_shell.zarr"
    )

    # Last init time was at least 2 hours ago and hour to 3-hour interval
    t0_datetime_utc = time_before_present(timedelta(hours=2)).floor(timedelta(hours=3))
    ds.init_time.values[:] = pd.date_range(
        t0_datetime_utc - timedelta(hours=3 * (len(ds.init_time) - 1)),
        t0_datetime_utc,
        freq=timedelta(hours=3),
    )

    # This is important to avoid saving errors
    for v in list(ds.coords.keys()):
        if ds.coords[v].dtype == object:
            ds[v].encoding.clear()

    for v in list(ds.variables.keys()):
        if ds[v].dtype == object:
            ds[v].encoding.clear()

    # Add data to dataset
    ds["UKV"] = xr.DataArray(
        np.zeros([len(ds[c]) for c in ds.coords]),
        coords=ds.coords,
    )

    # Add stored attributes to DataArray
    ds.UKV.attrs = ds.attrs["_data_attrs"]
    del ds.attrs["_data_attrs"]

    return ds


@pytest.fixture()
def sat_data():
    # Load dataset which only contains coordinates, but no data
    ds = xr.open_zarr(
        f"{os.path.dirname(os.path.abspath(__file__))}/test_data/sample_data/non_hrv_shell.zarr"
    )

    # Change times so they lead up to present. Delayed by at most 1 hour
    t0_datetime_utc = time_before_present(timedelta(minutes=0)).floor(timedelta(minutes=30))
    t0_datetime_utc = t0_datetime_utc - timedelta(minutes=30)
    ds.time.values[:] = pd.date_range(
        t0_datetime_utc - timedelta(minutes=5 * (len(ds.time) - 1)),
        t0_datetime_utc,
        freq=timedelta(minutes=5),
    )

    # Add data to dataset
    ds["data"] = xr.DataArray(
        np.zeros([len(ds[c]) for c in ds.coords]),
        coords=ds.coords,
    )

    # Add stored attributes to DataArray
    ds.data.attrs = ds.attrs["_data_attrs"]
    del ds.attrs["_data_attrs"]

    return ds


@pytest.fixture()
def sample_train_val_datamodule(multimodal_model):
    with tempfile.TemporaryDirectory() as tmpdirname:
        
        train_path = os.path.join(tmpdirname, "train")
        val_path = os.path.join(tmpdirname, "val")
        os.makedirs(train_path)
        os.makedirs(val_path)

        batch_size = 2
        forecast_len = multimodal_model.forecast_len
        history_len = multimodal_model.history_len
        seq_len = forecast_len + history_len + 1

        for file_n in range(2):
            sample = {
                "gsp": torch.randn(batch_size, seq_len),
                "gsp_id": torch.randint(0, 340, (batch_size, 1)),
                "gsp_time_utc": torch.randn(batch_size, seq_len),
                "solar_azimuth": torch.rand((batch_size, seq_len)),
                "solar_elevation": torch.rand((batch_size, seq_len)),
                "nwp": {"ukv": {"nwp": torch.rand((batch_size, 10, 7, 24, 24))}},
                "nwp_time_utc": torch.rand((batch_size, 7)),
                "nwp_target_time_utc": torch.rand((batch_size, 41)),
                'satellite_actual': torch.rand((batch_size, 7, 11, 24, 24)),
                'satellite_time_utc': torch.rand((batch_size, 7)),
            }
            torch.save(sample, os.path.join(train_path, f"train_{file_n:03}.pt"))

        for file_n in range(2):
            sample = {
                "gsp": torch.randn(batch_size, seq_len),
                "gsp_id": torch.randint(0, 340, (batch_size, 1)),
                "gsp_time_utc": torch.randn(batch_size, seq_len),
                "solar_azimuth": torch.rand((batch_size, seq_len)),
                "solar_elevation": torch.rand((batch_size, seq_len)),
                "nwp": {"ukv": {"nwp": torch.rand((batch_size, 10, 7, 24, 24))}},
                "nwp_time_utc": torch.rand((batch_size, 7)),
                "nwp_target_time_utc": torch.rand((batch_size, 41)),
                'satellite_actual': torch.rand((batch_size, 7, 11, 24, 24)),
                'satellite_time_utc': torch.rand((batch_size, 7)),
            }
            torch.save(sample, os.path.join(val_path, f"val_{file_n:03}.pt"))

        dm = DataModule(
            configuration=None,
            sample_dir=tmpdirname,
            batch_size=batch_size,
            num_workers=0,
            prefetch_factor=None,
            train_period=[None, None],
            val_period=[None, None],
        )
        return dm


@pytest.fixture()
def sample_datamodule():
    dm = DataModule(
        sample_dir="tests/test_data/presaved_samples_uk_regional",
        configuration=None,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
        train_period=[None, None],
        val_period=[None, None],
    )
    return dm


@pytest.fixture()
def sample_batch(sample_datamodule):
    batch = next(iter(sample_datamodule.train_dataloader()))
    
    # Add solar position data
    batch_size = batch["gsp"].shape[0]
    seq_len = batch["gsp"].shape[1]
    
    # Create solar position data
    if "solar_azimuth" not in batch or "solar_elevation" not in batch:
        batch["solar_azimuth"] = torch.rand((batch_size, seq_len))
        batch["solar_elevation"] = torch.rand((batch_size, seq_len))
    
    return batch


@pytest.fixture()
def sample_satellite_batch(sample_batch):
    sat_image = sample_batch["satellite_actual"]
    return torch.swapaxes(sat_image, 1, 2)


@pytest.fixture()
def sample_pv_batch():
    # TODO: Once PV site inputs are available from ocf-data-sampler UK regional remove these
    # old batches. For now we use the old batches to test the site encoder models
    return torch.load("tests/test_data/presaved_batches/train/000000.pt")


@pytest.fixture()
def sample_site_batch():
    dm = SiteDataModule(
        configuration=None,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
        train_period=[None, None],
        val_period=[None, None],
        sample_dir="tests/test_data/presaved_samples_site",
    )
    batch = next(iter(dm.train_dataloader()))
    return batch


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
        sequence_length=7,  # 30 minutes of 5 minutely satellite data = 7 time steps
        image_size_pixels=24,
        in_channels=11,
        out_features=128,
    )
    return kwargs


@pytest.fixture()
def site_encoder_model_kwargs():
    # Used to test site encoder model on PV data
    kwargs = dict(
        sequence_length=180 // 5 + 1,
        num_sites=349,
        out_features=128,
    )
    return kwargs


@pytest.fixture()
def site_encoder_model_kwargs_dsampler():
    # Used to test site encoder model on PV data
    kwargs = dict(
        sequence_length=60 // 15 + 1, num_sites=1, out_features=128, target_key_to_use="site"
    )
    return kwargs


@pytest.fixture()
def site_encoder_sensor_model_kwargs():
    # Used to test site encoder model on PV data
    kwargs = dict(
        sequence_length=180 // 5 + 1,
        num_sites=26,
        out_features=128,
        num_channels=23,
        target_key_to_use="wind",
        input_key_to_use="sensor",
    )
    return kwargs


@pytest.fixture()
def raw_multimodal_model_kwargs(model_minutes_kwargs):
    kwargs = dict(
        sat_encoder=dict(
            _target_="pvnet.models.multimodal.encoders.encoders3d.DefaultPVNet",
            _partial_=True,
            in_channels=11,
            out_features=128,
            number_of_conv3d_layers=6,
            conv3d_channels=32,
            image_size_pixels=24,
        ),
        nwp_encoders_dict={
            "ukv": dict(
                _target_="pvnet.models.multimodal.encoders.encoders3d.DefaultPVNet",
                _partial_=True,
                in_channels=11,
                out_features=128,
                number_of_conv3d_layers=6,
                conv3d_channels=32,
                image_size_pixels=24,
            ),
        },
        add_image_embedding_channel=True,
        # ocf-data-sampler doesn't supprt PV site inputs yet
        pv_encoder=None,
        output_network=dict(
            _target_="pvnet.models.multimodal.linear_networks.networks.ResFCNet2",
            _partial_=True,
            fc_hidden_features=128,
            n_res_blocks=6,
            res_block_layers=2,
            dropout_frac=0.0,
        ),
        embedding_dim=16,
        include_sun=True,
        include_gsp_yield_history=True,
        sat_history_minutes=30,
        nwp_history_minutes={"ukv": 120},
        nwp_forecast_minutes={"ukv": 480},
        min_sat_delay_minutes=0,
    )

    kwargs.update(model_minutes_kwargs)

    return kwargs


@pytest.fixture()
def multimodal_model_kwargs(raw_multimodal_model_kwargs):
    return hydra.utils.instantiate(raw_multimodal_model_kwargs)


@pytest.fixture()
def multimodal_model(multimodal_model_kwargs):
    model = Model(**multimodal_model_kwargs)
    return model


@pytest.fixture()
def multimodal_quantile_model(multimodal_model_kwargs):
    model = Model(output_quantiles=[0.1, 0.5, 0.9], **multimodal_model_kwargs)
    return model


@pytest.fixture()
def multimodal_quantile_model_ignore_minutes(multimodal_model_kwargs):
    """Only forecsat second half of the 8 hours"""
    model = Model(
        output_quantiles=[0.1, 0.5, 0.9], **multimodal_model_kwargs, forecast_minutes_ignore=240
    )
    return model
