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

from ocf_datapipes.batch import BatchKey

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


def generate_synthetic_sample():
    """
    Generate a synthetic sample contrary to utilising reference .pt files
    """
    now = pd.Timestamp.now(tz='UTC')    
    sample = {}
    
    # NWP define
    sample["nwp"] = {
        "ukv": {
            "nwp": torch.rand(11, 11, 24, 24),
            "nwp_init_time_utc": torch.tensor(
                [(now - pd.Timedelta(hours=i)).timestamp() for i in range(11)]
            ),
            "nwp_step": torch.arange(11, dtype=torch.float32),
            "nwp_target_time_utc": torch.tensor(
                [(now + pd.Timedelta(hours=i)).timestamp() for i in range(11)]
            ),
            "nwp_y_osgb": torch.linspace(0, 100, 24),
            "nwp_x_osgb": torch.linspace(0, 100, 24),
        },
        "ecmwf": {
            "nwp": torch.rand(11, 12, 12, 12),
            "nwp_init_time_utc": torch.tensor(
                [(now - pd.Timedelta(hours=i)).timestamp() for i in range(11)]
            ),
            "nwp_step": torch.arange(11, dtype=torch.float32),
            "nwp_target_time_utc": torch.tensor(
                [(now + pd.Timedelta(hours=i)).timestamp() for i in range(11)]
            ),
        },
        "sat_pred": {
            "nwp": torch.rand(12, 11, 24, 24),
            "nwp_init_time_utc": torch.tensor(
                [(now - pd.Timedelta(hours=i)).timestamp() for i in range(12)]
            ),
            "nwp_step": torch.arange(12, dtype=torch.float32),
            "nwp_target_time_utc": torch.tensor(
                [(now + pd.Timedelta(hours=i)).timestamp() for i in range(12)]
            ),
        },
    }
    
    # Satellite define
    sample["satellite_actual"] = torch.rand(7, 11, 24, 24)
    sample["satellite_time_utc"] = torch.tensor(
        [(now - pd.Timedelta(minutes=5*i)).timestamp() for i in range(7)]
    )
    sample["satellite_x_geostationary"] = torch.linspace(0, 100, 24)
    sample["satellite_y_geostationary"] = torch.linspace(0, 100, 24)
    
    # GSP define
    sample["gsp"] = torch.rand(21)
    sample["gsp_nominal_capacity_mwp"] = torch.tensor(100.0)
    sample["gsp_effective_capacity_mwp"] = torch.tensor(85.0)
    sample["gsp_time_utc"] = torch.tensor(
        [(now + pd.Timedelta(minutes=30*i)).timestamp() for i in range(21)]
    )
    sample["gsp_t0_idx"] = float(7)
    sample["gsp_id"] = 12
    sample["gsp_x_osgb"] = 123456.0
    sample["gsp_y_osgb"] = 654321.0
    
    # Solar position define
    sample["solar_azimuth"] = torch.linspace(0, 180, 21)
    sample["solar_elevation"] = torch.linspace(-10, 60, 21)
    
    return sample


def generate_synthetic_site_sample():
    """
    Generate synthetic site sample
    """
    now = pd.Timestamp.now(tz='UTC')
    
    sample = {
        "site": torch.rand(5, 1),
        "time_utc": torch.tensor(
            [(now - pd.Timedelta(minutes=15*i)).timestamp() for i in range(5)]
        ),
        "site_id": torch.tensor(1),
        "x_osgb": torch.tensor(100.0),
        "y_osgb": torch.tensor(300.0),
        "capacity_mwp": torch.tensor(10.0),
    }
    
    return sample


def generate_synthetic_pv_batch():
    """
    Generate a synthetic PV batch for SimpleLearnedAggregator tests
    """
    # 3D tensor of shape [batch_size, sequence_length, num_sites]
    batch_size = 8
    sequence_length = 180 // 5 + 1
    num_sites = 349    

    return torch.rand(batch_size, sequence_length, num_sites)


@pytest.fixture()
def sample_train_val_datamodule():
    """
    Create a DataModule with synthetic data files for training and validation
    """
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create train and val directories
        os.makedirs(f"{tmpdirname}/train", exist_ok=True)
        os.makedirs(f"{tmpdirname}/val", exist_ok=True)
        
        # Generate and save synthetic samples
        base_sample = generate_synthetic_sample()
        
        for i in range(10):
            # Create modified copy for each sample
            sample = {}
            for key, value in base_sample.items():
                if isinstance(value, dict):
                    sample[key] = {}
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, dict):
                            sample[key][subkey] = {}
                            for subsubkey, subsubvalue in subvalue.items():
                                if isinstance(subsubvalue, torch.Tensor) and subsubvalue.dtype.is_floating_point:
                                    sample[key][subkey][subsubkey] = subsubvalue + torch.randn_like(subsubvalue) * 0.01
                                else:
                                    sample[key][subkey][subsubkey] = subsubvalue
                        elif isinstance(subvalue, torch.Tensor) and subvalue.dtype.is_floating_point:
                            sample[key][subkey] = subvalue + torch.randn_like(subvalue) * 0.01
                        else:
                            sample[key][subkey] = subvalue
                elif isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
                    sample[key] = value + torch.randn_like(value) * 0.01
                else:
                    sample[key] = value
            
            # Save for both train and val
            torch.save(sample, f"{tmpdirname}/train/{i:08d}.pt")
            torch.save(sample, f"{tmpdirname}/val/{i:08d}.pt")
        
        # Define DataModule with temporary directory
        dm = DataModule(
            configuration=None,
            sample_dir=tmpdirname,
            batch_size=2,
            num_workers=0,
            prefetch_factor=None,
            train_period=[None, None],
            val_period=[None, None],
        )
        
        yield dm


@pytest.fixture()
def sample_datamodule():
    """
    Create a DataModule with synthetic data files
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create train and val directories
        os.makedirs(f"{tmpdirname}/train", exist_ok=True)
        os.makedirs(f"{tmpdirname}/val", exist_ok=True)
        
        # Generate and save synthetic samples
        for i in range(10):
            sample = generate_synthetic_sample()
            torch.save(sample, f"{tmpdirname}/train/{i:08d}.pt")
            torch.save(sample, f"{tmpdirname}/val/{i:08d}.pt")
        
        # Define DataModule with temporary directory
        dm = DataModule(
            configuration=None,
            sample_dir=tmpdirname,
            batch_size=2,
            num_workers=0,
            prefetch_factor=None,
            train_period=[None, None],
            val_period=[None, None],
        )
        
        yield dm


@pytest.fixture()
def sample_site_datamodule():
    """
    Create a SiteDataModule with synthetic site data files
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create train and val directories
        os.makedirs(f"{tmpdirname}/train", exist_ok=True)
        os.makedirs(f"{tmpdirname}/val", exist_ok=True)
        
        # Generate and save synthetic site samples
        for i in range(10):
            sample = generate_synthetic_site_sample()
            torch.save(sample, f"{tmpdirname}/train/{i:08d}.pt")
            torch.save(sample, f"{tmpdirname}/val/{i:08d}.pt")
        
        # Define SiteDataModule with temporary directory
        dm = SiteDataModule(
            configuration=None,
            sample_dir=tmpdirname,
            batch_size=2,
            num_workers=0,
            prefetch_factor=None,
            train_period=[None, None],
            val_period=[None, None],
        )
        
        yield dm


@pytest.fixture()
def sample_batch(sample_datamodule):
    batch = next(iter(sample_datamodule.train_dataloader()))
    return batch


@pytest.fixture()
def sample_satellite_batch(sample_batch):
    sat_image = sample_batch["satellite_actual"]
    return torch.swapaxes(sat_image, 1, 2)


@pytest.fixture()
def sample_pv_batch():
    """
    Currently overrides utilising reference .pt for updated gsp_id and pv
    Intermediate change
    """

    # TODO: Once PV site inputs are available from ocf-data-sampler UK regional remove these
    # old batches. For now we use the old batches to test the site encoder models

    file_path = "tests/test_data/presaved_batches/train/000000.pt"
    old_batch = torch.load(file_path)
    new_batch = {}

    for key, value in old_batch.items():
        if key == BatchKey.pv:
            new_batch["pv"] = value
            key_pv_found = True
        elif key == BatchKey.gsp_id:
            new_batch["gsp_id"] = value
            key_gsp_id_found = True
        else:
            new_batch[key] = value

    return new_batch


# @pytest.fixture()
# def sample_site_batch():
#     """
#     Create a batch of site data for SingleAttentionNetwork tests    
#     """
#     site_tensor = torch.rand(2, 5, 1)    
#     return {
#         "site": site_tensor,
#         "site_id": torch.tensor([1, 2]),
#     }


@pytest.fixture()
def sample_site_batch(sample_site_datamodule):
    batch = next(iter(sample_site_datamodule.train_dataloader()))
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
