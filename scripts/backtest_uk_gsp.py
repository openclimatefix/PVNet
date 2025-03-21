"""
A script to run backtest for PVNet and the summation model for UK regional and national

Use:

- This script uses hydra to construct the config, just like in `run.py`. So you need to make sure
  that the data config is set up appropriate for the model being run in this script
- The PVNet and summation model checkpoints; the time range over which to make predictions are made;
  and the output directory where the results near the top of the script as hard coded user
  variables. These should be changed.


```
python backtest_uk_gsp.py
```

"""

try:
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass

import logging
import os
import sys

import hydra
import numpy as np
import pandas as pd
import torch
import xarray as xr
from ocf_data_sampler.sample.base import batch_to_tensor, copy_batch_to_device
from ocf_datapipes.batch import (
    BatchKey,
    NumpyBatch,
)
from ocf_datapipes.config.load import load_yaml_configuration
from ocf_datapipes.load import OpenGSP
from ocf_datapipes.training.common import _get_datapipes_dict
from ocf_datapipes.training.pvnet_all_gsp import construct_sliced_data_pipeline, create_t0_datapipe
from ocf_datapipes.utils.consts import ELEVATION_MEAN, ELEVATION_STD
from omegaconf import DictConfig

# TODO: Having this script rely on pvnet_app sets up a circular dependency. The function
# `preds_to_dataarray()` should probably be moved here
from pvnet_app.utils import preds_to_dataarray
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter import IterableWrapper
from tqdm import tqdm

from pvnet.load_model import get_model_from_checkpoints

# ------------------------------------------------------------------
# USER CONFIGURED VARIABLES
output_dir = "/mnt/disks/extra_batches/test_backtest"

# Local directory to load the PVNet checkpoint from. By default this should pull the best performing
# checkpoint on the val set
model_chckpoint_dir = "/home/jamesfulton/repos/PVNet/checkpoints/q911tei5"

# Local directory to load the summation model checkpoint from. By default this should pull the best
# performing checkpoint on the val set. If set to None a simple sum is used instead
summation_chckpoint_dir = (
    "/home/jamesfulton/repos/PVNet_summation/checkpoints/pvnet_summation/73oa4w9t"
)

# Forecasts will be made for all available init times between these
start_datetime = "2022-05-08 00:00"
end_datetime = "2022-05-08 00:30"

# ------------------------------------------------------------------
# SET UP LOGGING

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# ------------------------------------------------------------------
# DERIVED VARIABLES

# This will run on GPU if it exists
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# GLOBAL VARIABLES

# The frequency of the GSP data
FREQ_MINS = 30

# When sun as elevation below this, the forecast is set to zero
MIN_DAY_ELEVATION = 0

# All regional GSP IDs - not including national which is treated separately
ALL_GSP_IDS = np.arange(1, 318)

# ------------------------------------------------------------------
# FUNCTIONS


def get_gsp_ds(config_path: str) -> xr.Dataset:
    """Load GSP data from the path in the data config.

    Args:
        config_path: Path to the data configuration file

    Returns:
        xarray.Dataset of PVLive truths and capacities
    """

    config = load_yaml_configuration(config_path)
    gsp_datapipe = OpenGSP(gsp_pv_power_zarr_path=config.input_data.gsp.gsp_zarr_path)
    ds_gsp = next(iter(gsp_datapipe))

    return ds_gsp


def get_available_t0_times(start_datetime, end_datetime, config_path):
    """Filter a list of t0 init-times to those for which all required input data is available.

    Args:
        start_datetime: First potential t0 time
        end_datetime: Last potential t0 time
        config_path: Path to data config file

    Returns:
        pandas.DatetimeIndex of the init-times available for required inputs
    """

    start_datetime = pd.Timestamp(start_datetime)
    end_datetime = pd.Timestamp(end_datetime)
    # Open all the input data so we can check what of the potential data init times we have input
    # data for
    datapipes_dict = _get_datapipes_dict(config_path, production=False)

    # Pop out the config file
    config = datapipes_dict.pop("config")

    # We are going to abuse the `create_t0_datapipe()` function to find the init-times in
    # potential_init_times which we have input data for. To do this, we will feed in some fake GSP
    # data which has the potential_init_times as timestamps. This is a bit hacky but works for now

    # Set up init-times we would like to make predictions for
    potential_init_times = pd.date_range(start_datetime, end_datetime, freq=f"{FREQ_MINS}min")

    # We buffer the potential init-times so that we don't lose any init-times from the
    # start and end. Again this is a hacky step
    history_duration = pd.Timedelta(config.input_data.gsp.history_minutes, "min")
    forecast_duration = pd.Timedelta(config.input_data.gsp.forecast_minutes, "min")
    buffered_potential_init_times = pd.date_range(
        start_datetime - history_duration, end_datetime + forecast_duration, freq=f"{FREQ_MINS}min"
    )

    ds_fake_gsp = buffered_potential_init_times.to_frame().to_xarray().rename({"index": "time_utc"})
    ds_fake_gsp = ds_fake_gsp.rename({0: "gsp_pv_power_mw"})
    ds_fake_gsp = ds_fake_gsp.expand_dims("gsp_id", axis=1)
    ds_fake_gsp = ds_fake_gsp.assign_coords(
        gsp_id=[0],
        x_osgb=("gsp_id", [0]),
        y_osgb=("gsp_id", [0]),
    )
    ds_fake_gsp = ds_fake_gsp.gsp_pv_power_mw.astype(float) * 1e-18

    # Overwrite the GSP data which is already in the datapipes dict
    datapipes_dict["gsp"] = IterableWrapper([ds_fake_gsp])

    # Use create_t0_datapipe to get datapipe of init-times
    t0_datapipe = create_t0_datapipe(
        datapipes_dict,
        configuration=config,
        shuffle=False,
    )

    # Create a full list of available init-times
    available_init_times = pd.to_datetime([t0 for t0 in t0_datapipe])

    logger.info(
        f"{len(available_init_times)} out of {len(potential_init_times)} "
        "requested init-times have required input data"
    )

    return available_init_times


def get_times_datapipe(config_path):
    """Create init-time datapipe

    Args:
        config_path: Path to data config file

    Returns:
        Datapipe: A Datapipe yielding init-times
    """

    # Filter the init-times to times we have all input data for
    available_target_times = get_available_t0_times(
        start_datetime,
        end_datetime,
        config_path,
    )
    num_t0s = len(available_target_times)

    # Save the init-times which predictions are being made for. This is really helpful to check
    # whilst the backtest is running since it takes a long time. This lets you see what init-times
    # the backtest will end up producing
    available_target_times.to_frame().to_csv(f"{output_dir}/t0_times.csv")

    # Create times datapipe so each worker receives 317 copies of the same datetime for its batch
    t0_datapipe = IterableWrapper(available_target_times)
    t0_datapipe = t0_datapipe.sharding_filter()

    t0_datapipe = t0_datapipe.set_length(num_t0s)

    return t0_datapipe


class ModelPipe:
    """A class to conveniently make and process predictions from batches"""

    def __init__(self, model, summation_model, ds_gsp: xr.Dataset):
        """A class to conveniently make and process predictions from batches

        Args:
            model: PVNet GSP level model
            summation_model: Summation model to make national forecast from GSP level forecasts
            ds_gsp:xarray dataset of PVLive true values and capacities
        """
        self.model = model
        self.summation_model = summation_model
        self.ds_gsp = ds_gsp

    def predict_batch(self, batch: NumpyBatch) -> xr.Dataset:
        """Run the batch through the model and compile the predictions into an xarray DataArray

        Args:
            batch: A batch of samples with inputs for each GSP for the same init-time

        Returns:
            xarray.Dataset of all GSP and national forecasts for the batch
        """

        # Unpack some variables from the batch
        id0 = batch[BatchKey.gsp_t0_idx]
        t0 = batch[BatchKey.gsp_time_utc].cpu().numpy().astype("datetime64[s]")[0, id0]
        n_valid_times = len(batch[BatchKey.gsp_time_utc][0, id0 + 1 :])
        ds_gsp = self.ds_gsp
        model = self.model
        summation_model = self.summation_model

        # Get valid times for this forecast
        valid_times = pd.to_datetime(
            [t0 + np.timedelta64((i + 1) * FREQ_MINS, "m") for i in range(n_valid_times)]
        )

        # Get effective capacities for this forecast
        gsp_capacities = ds_gsp.effective_capacity_mwp.sel(
            time_utc=t0, gsp_id=slice(1, None)
        ).values
        national_capacity = ds_gsp.effective_capacity_mwp.sel(time_utc=t0, gsp_id=0).item()

        # Get the solar elevations. We need to un-normalise these from the values in the batch
        elevation = batch[BatchKey.gsp_solar_elevation] * ELEVATION_STD + ELEVATION_MEAN
        # We only need elevation mask for forecasted values, not history
        elevation = elevation[:, id0 + 1 :]

        # Make mask dataset for sundown
        da_sundown_mask = xr.DataArray(
            data=elevation < MIN_DAY_ELEVATION,
            dims=["gsp_id", "target_datetime_utc"],
            coords=dict(
                gsp_id=ALL_GSP_IDS,
                target_datetime_utc=valid_times,
            ),
        )

        with torch.no_grad():
            # Run batch through model to get 0-1 predictions for all GSPs
            device_batch = copy_batch_to_device(batch_to_tensor(batch), device)
            y_normed_gsp = model(device_batch).detach().cpu().numpy()

        da_normed_gsp = preds_to_dataarray(y_normed_gsp, model, valid_times, ALL_GSP_IDS)

        # Multiply normalised forecasts by capacities and clip negatives
        da_abs_gsp = da_normed_gsp.clip(0, None) * gsp_capacities[:, None, None]

        # Apply sundown mask
        da_abs_gsp = da_abs_gsp.where(~da_sundown_mask).fillna(0.0)

        # Make national predictions using summation model
        if summation_model is not None:
            with torch.no_grad():
                # Construct sample for the summation model
                summation_inputs = {
                    "pvnet_outputs": torch.Tensor(y_normed_gsp[np.newaxis]).to(device),
                    "effective_capacity": (
                        torch.Tensor(gsp_capacities / national_capacity)
                        .to(device)
                        .unsqueeze(0)
                        .unsqueeze(-1)
                    ),
                }

                # Run batch through the summation model
                y_normed_national = (
                    summation_model(summation_inputs).detach().squeeze().cpu().numpy()
                )

            # Convert national predictions to DataArray
            da_normed_national = preds_to_dataarray(
                y_normed_national[np.newaxis], summation_model, valid_times, gsp_ids=[0]
            )

            # Multiply normalised forecasts by capacities and clip negatives
            da_abs_national = da_normed_national.clip(0, None) * national_capacity

            # Apply sundown mask - All GSPs must be masked to mask national
            da_abs_national = da_abs_national.where(~da_sundown_mask.all(dim="gsp_id")).fillna(0.0)

        # If no summation model, make national predictions using simple sum
        else:
            da_abs_national = (
                da_abs_gsp.sum(dim="gsp_id")
                .expand_dims(dim="gsp_id", axis=0)
                .assign_coords(gsp_id=[0])
            )

        # Concat the regional GSP and national predictions
        da_abs_all = xr.concat([da_abs_national, da_abs_gsp], dim="gsp_id")
        ds_abs_all = da_abs_all.to_dataset(name="hindcast")

        ds_abs_all = ds_abs_all.expand_dims(dim="init_time_utc", axis=0).assign_coords(
            init_time_utc=[t0]
        )

        return ds_abs_all


def get_datapipe(config_path: str) -> NumpyBatch:
    """Construct datapipe yielding batches of concurrent samples for all GSPs

    Args:
        config_path: Path to the data configuration file

    Returns:
        NumpyBatch: Concurrent batch of samples for each GSP
    """

    # Construct location and init-time datapipes
    t0_datapipe = get_times_datapipe(config_path)

    # Construct sample datapipes
    data_pipeline = construct_sliced_data_pipeline(
        config_path,
        t0_datapipe,
    )

    # Convert to tensor for model
    data_pipeline = data_pipeline.map(batch_to_tensor).set_length(len(t0_datapipe))

    return data_pipeline


@hydra.main(config_path="../configs", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig):
    """Runs the backtest"""

    dataloader_kwargs = dict(
        shuffle=False,
        batch_size=None,
        sampler=None,
        batch_sampler=None,
        # Number of workers set in the config file
        num_workers=config.datamodule.num_workers,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        prefetch_factor=config.datamodule.prefetch_factor,
        persistent_workers=False,
    )

    # Set up output dir
    os.makedirs(output_dir)

    # Create concurrent batch datapipe
    # Each batch includes a sample for each of the 317 GSPs for a single init-time
    batch_pipe = get_datapipe(config.datamodule.configuration)
    num_batches = len(batch_pipe)

    # Load the GSP data as an xarray object
    ds_gsp = get_gsp_ds(config.datamodule.configuration)

    # Create a dataloader for the concurrent batches and use multiprocessing
    dataloader = DataLoader(batch_pipe, **dataloader_kwargs)

    # Load the PVNet model and summation model
    model, *_ = get_model_from_checkpoints([model_chckpoint_dir], val_best=True)
    model = model.eval().to(device)
    if summation_chckpoint_dir is None:
        summation_model = None
    else:
        summation_model, *_ = get_model_from_checkpoints([summation_chckpoint_dir], val_best=True)
        summation_model = summation_model.eval().to(device)

    # Create object to make predictions for each input batch
    model_pipe = ModelPipe(model, summation_model, ds_gsp)

    # Loop through the batches
    pbar = tqdm(total=num_batches)
    for i, batch in zip(range(num_batches), dataloader):
        # Make predictions for the init-time
        ds_abs_all = model_pipe.predict_batch(batch)

        t0 = ds_abs_all.init_time_utc.values[0]

        # Save the predictioons
        filename = f"{output_dir}/{t0}.nc"
        ds_abs_all.to_netcdf(filename)

        pbar.update()

    # Close down
    pbar.close()
    del dataloader


if __name__ == "__main__":
    main()
