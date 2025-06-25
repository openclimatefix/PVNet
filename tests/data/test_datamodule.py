import os
import tempfile
import pytest
import torch
import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import OmegaConf
from unittest.mock import patch

from pvnet.data import DataModule, SiteDataModule


@pytest.fixture
def temp_pt_sample_dir():
    """Create temporary directory with synthetic PT samples"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create train and val directories
        os.makedirs(f"{tmpdirname}/train", exist_ok=True)
        os.makedirs(f"{tmpdirname}/val", exist_ok=True)

        # Generate and save synthetic samples
        for i in range(5):
            sample = {
                "gsp": torch.rand(21),
                "gsp_time_utc": torch.tensor(list(range(21))),
                "gsp_nominal_capacity_mwp": torch.tensor(100.0),
                "gsp_id": 12
            }
            torch.save(sample, f"{tmpdirname}/train/{i:08d}.pt")
            torch.save(sample, f"{tmpdirname}/val/{i:08d}.pt")

        yield tmpdirname


@pytest.fixture
def temp_nc_sample_dir():
    """Create temporary directory with synthetic NC site samples"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create train and val directories
        os.makedirs(f"{tmpdirname}/train", exist_ok=True)
        os.makedirs(f"{tmpdirname}/val", exist_ok=True)

        # Create config file
        config_path = f"{tmpdirname}/data_configuration.yaml"
        with open(config_path, "w") as f:
            f.write(f"sample_dir: {tmpdirname}\n")

        # Generate and save synthetic site samples
        for i in range(5):
            site_time = pd.date_range("2023-01-01", periods=10, freq="15min")
            ds = xr.Dataset(
                data_vars={
                    "site": (["site__time_utc"], np.random.rand(10)),
                },
                coords={
                    "site__time_utc": site_time,
                    "site__site_id": np.int32(i % 3 + 1),
                    "site__latitude": 52.5,
                    "site__longitude": -1.5,
                    "site__capacity_kwp": 10000.0,
                }
            )

            ds.to_netcdf(f"{tmpdirname}/train/{i:08d}.nc", mode="w", engine="h5netcdf")
            ds.to_netcdf(f"{tmpdirname}/val/{i:08d}.nc", mode="w", engine="h5netcdf")

        yield tmpdirname


def test_init(temp_pt_sample_dir):
    """Test DataModule initialization"""
    dm = DataModule(
        configuration=None,
        sample_dir=temp_pt_sample_dir,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
        train_period=[None, None],
        val_period=[None, None],
    )

    # Verify datamodule initialisation
    assert dm is not None
    assert hasattr(dm, "train_dataloader")


def test_iter(temp_pt_sample_dir):
    """Test iteration through DataModule"""
    dm = DataModule(
        configuration=None,
        sample_dir=temp_pt_sample_dir,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
        train_period=[None, None],
        val_period=[None, None],
    )

    # Verify existing keys
    batch = next(iter(dm.train_dataloader()))
    assert batch is not None
    assert "gsp" in batch


def test_iter_multiprocessing(temp_pt_sample_dir):
    """Test DataModule with multiple workers"""
    dm = DataModule(
        configuration=None,
        sample_dir=temp_pt_sample_dir,
        batch_size=1,
        num_workers=2,
        prefetch_factor=1,
        train_period=[None, None],
        val_period=[None, None],
    )

    served_batches = 0
    for batch in dm.train_dataloader():
        served_batches += 1

        if served_batches == 2:
            break

    # Batch verification
    assert served_batches == 2


def test_site_init_sample_dir(temp_nc_sample_dir):
    """Test SiteDataModule initialization with sample dir"""
    dm = SiteDataModule(
        configuration=None,
        sample_dir=temp_nc_sample_dir,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
        train_period=[None, None],
        val_period=[None, None],
    )

    # Verify datamodule initialisation
    assert dm is not None
    assert hasattr(dm, "train_dataloader")


def test_site_init_config(temp_nc_sample_dir):
    """Test SiteDataModule initialization with config file"""
    config_path = f"{temp_nc_sample_dir}/data_configuration.yaml"

    dm = SiteDataModule(
        configuration=config_path,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
        train_period=[None, None],
        val_period=[None, None],
        sample_dir=None,
    )

    # Verify datamodule initialisation w/ config
    assert dm is not None
    assert hasattr(dm, "train_dataloader")


def test_hardcoded_augmentations_applied_on_premade_samples():
    """Test that the hard-coded augmentations are applied to pre-made samples."""
    # Define sizes for synthetic image-like data
    nwp_channels, nwp_height, nwp_width = 5, 16, 16
    sat_channels, sat_height, sat_width = 12, 32, 32

    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create a temporary directory for pre-made samples
        train_dir = f"{tmpdirname}/train"
        os.makedirs(train_dir, exist_ok=True)
        
        # We need a class that matches the one used in the datamodule
        # It needs `load` and `to_numpy` methods for the test
        class DummySample:
            def __init__(self, data):
                self.data = data
            
            @classmethod
            def load(cls, path):
                return cls(torch.load(path, map_location='cpu', weights_only=False))

            def to_numpy(self):
                # Convert torch tensors in the sample to numpy arrays
                numpy_sample = {}
                for key, value in self.data.items():
                    if isinstance(value, torch.Tensor):
                        numpy_sample[key] = value.numpy()
                    elif isinstance(value, dict):
                         numpy_sample[key] = {
                             k: v.numpy() if isinstance(v, torch.Tensor) else v
                             for k, v in value.items()
                         }
                    else:
                        numpy_sample[key] = value
                return numpy_sample

        # Create an original sample file to compare against
        original_sample_dict = {
            "gsp_id": 12,
            "nwp": {
                "ecmwf": {
                    "data": np.random.rand(nwp_channels, nwp_height, nwp_width).astype(np.float32)
                }
            },
            "satellite": np.random.rand(sat_channels, sat_height, sat_width).astype(np.float32)
        }
        torch.save(original_sample_dict, f"{train_dir}/00000000.pt")

        # The PremadeSamplesDataset needs to be patched to use our DummySample class
        with patch('pvnet.data.uk_regional_datamodule.UKRegionalSample', new=DummySample):
        
            # Instantiate DataModule to use the pre-made samples path.
            # No augmentation config is needed as it's hard-coded.
            dm = DataModule(
                configuration=None,
                sample_dir=tmpdirname,
                batch_size=1,
                num_workers=0,
            )

            # Get one batch from the training dataloader
            augmented_batch = next(iter(dm.train_dataloader()))

            # Extract tensors for comparison
            augmented_nwp = augmented_batch["nwp"]["ecmwf"]["data"].squeeze(0)
            augmented_satellite = augmented_batch["satellite"].squeeze(0)

            original_nwp = torch.from_numpy(original_sample_dict["nwp"]["ecmwf"]["data"])
            original_satellite = torch.from_numpy(original_sample_dict["satellite"])

            # Assert that the augmentations have changed the data
            assert not torch.equal(augmented_nwp, original_nwp)
            assert not torch.equal(augmented_satellite, original_satellite)

            # Assert that the shapes are still the same
            assert augmented_nwp.shape == original_nwp.shape
            assert augmented_satellite.shape == original_satellite.shape
