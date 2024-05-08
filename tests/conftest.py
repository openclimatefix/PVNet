import os
import glob
import tempfile

import pytest
import pandas as pd
import numpy as np
import xarray as xr
import torch
import hydra

from ocf_datapipes.batch import BatchKey
from datetime import timedelta

import pvnet
from pvnet.data.datamodule import DataModule
from pvnet.data.wind_datamodule import WindDataModule

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
def sample_train_val_datamodule():
    # duplicate the sample batcnes for more training/val data
    n_duplicates = 10

    with tempfile.TemporaryDirectory() as tmpdirname:
        os.makedirs(f"{tmpdirname}/train")
        os.makedirs(f"{tmpdirname}/val")

        file_n = 0

        for file in glob.glob("tests/test_data/sample_batches/train/*.pt"):
            batch = torch.load(file)

            for i in range(n_duplicates):
                # Save fopr both train and val
                torch.save(batch, f"{tmpdirname}/train/{file_n:06}.pt")
                torch.save(batch, f"{tmpdirname}/val/{file_n:06}.pt")

                file_n += 1

        dm = DataModule(
            configuration=None,
            batch_size=2,
            num_workers=0,
            prefetch_factor=None,
            train_period=[None, None],
            val_period=[None, None],
            test_period=[None, None],
            batch_dir=f"{tmpdirname}",
        )
        yield dm


@pytest.fixture()
def sample_datamodule():
    dm = DataModule(
        configuration=None,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
        train_period=[None, None],
        val_period=[None, None],
        test_period=[None, None],
        batch_dir="tests/test_data/sample_batches",
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
def sample_pv_batch(sample_batch):
    pv_data = sample_batch[BatchKey.pv]
    return pv_data


@pytest.fixture()
def sample_wind_batch():
    dm = WindDataModule(
        configuration=None,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
        train_period=[None, None],
        val_period=[None, None],
        test_period=[None, None],
        batch_dir="tests/test_data/sample_wind_batches",
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
        sequence_length=(90 - 30) // 5 + 1,
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
            _target_=pvnet.models.multimodal.encoders.encoders3d.DefaultPVNet,
            _partial_=True,
            in_channels=11,
            out_features=128,
            number_of_conv3d_layers=6,
            conv3d_channels=32,
            image_size_pixels=24,
        ),
        nwp_encoders_dict={
            "ukv": dict(
                _target_=pvnet.models.multimodal.encoders.encoders3d.DefaultPVNet,
                _partial_=True,
                in_channels=2,
                out_features=128,
                number_of_conv3d_layers=6,
                conv3d_channels=32,
                image_size_pixels=24,
            ),
        },
        add_image_embedding_channel=True,
        pv_encoder=dict(
            _target_=pvnet.models.multimodal.site_encoders.encoders.SingleAttentionNetwork,
            _partial_=True,
            num_sites=349,
            out_features=40,
            num_heads=4,
            kdim=40,
            id_embed_dim=20,
        ),
        output_network=dict(
            _target_=pvnet.models.multimodal.linear_networks.networks.ResFCNet2,
            _partial_=True,
            fc_hidden_features=128,
            n_res_blocks=6,
            res_block_layers=2,
            dropout_frac=0.0,
        ),
        embedding_dim=16,
        include_sun=True,
        include_gsp_yield_history=True,
        sat_history_minutes=90,
        nwp_history_minutes={"ukv": 120},
        nwp_forecast_minutes={"ukv": 480},
        pv_history_minutes=180,
        min_sat_delay_minutes=30,
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
def multimodal_weighted_quantile_model(multimodal_model_kwargs):
    model = Model(
        output_quantiles=[0.1, 0.5, 0.9], **multimodal_model_kwargs, use_weighted_loss=True
    )
    return model
