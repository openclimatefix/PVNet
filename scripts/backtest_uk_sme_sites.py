"""
A script to run backtest for PVNetfor SME sites in the UK

Use:

- This script uses hydra to construct the config, just like in `run.py`. So you need to make sure
  that the data config is set up appropriate for the model being run in this script
- The PVNet checkpoint; the time range over which to make predictions are made;
  and the output directory where the results near the top of the script as hard coded user
  variables. These should be changed.


```
python backtest.py
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
from ocf_datapipes.batch import (
    BatchKey,
    NumpyBatch,
    batch_to_tensor,
    copy_batch_to_device,
    stack_np_examples_into_batch,
)
from ocf_datapipes.config.load import load_yaml_configuration
from ocf_datapipes.load.pv.pv import OpenPVFromNetCDFIterDataPipe
from ocf_datapipes.training.common import create_t0_and_loc_datapipes
from ocf_datapipes.training.pvnet_site import (_get_datapipes_dict, construct_sliced_data_pipeline, DictDatasetIterDataPipe, ConvertToNumpyBatchIterDataPipe, split_dataset_dict_dp)
from ocf_datapipes.utils.consts import ELEVATION_MEAN, ELEVATION_STD
from omegaconf import DictConfig
from ocf_datapipes.utils.utils import combine_to_single_dataset

from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter import IterableWrapper
from tqdm import tqdm

from pvnet.load_model import get_model_from_checkpoints
from pvnet.utils import SiteLocationLookup

# ------------------------------------------------------------------
# USER CONFIGURED VARIABLES
output_dir = "/home/sukhil/ocf-code/PVNet/backtest/test_backtest"

# Local directory to load the PVNet checkpoint from. By default this should pull the best performing
# checkpoint on the val set
model_chckpoint_dir = "/home/sukhil/ocf-code/PVNet/PLACEHOLDER/ggu4in5x"

# Local directory to load the summation model checkpoint from. By default this should pull the best
# performing checkpoint on the val set. If set to None a simple sum is used instead
# summation_chckpoint_dir = (
#     "/home/jamesfulton/repos/PVNet_summation/checkpoints/pvnet_summation/nw673nw2"
# )

# Forecasts will be made for all available init times between these
start_datetime = "2022-07-08 09:00"
end_datetime = "2022-07-08 13:00"

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
ALL_SITE_ML_IDS = [ 4353,  6670,  4403,  6331,  6332,  6590,  4416,  4417,  6592,
        6594,  6596,  4423,  6600,  6605, 27855,  6378,  6381,  6385,
       25842]

# ------------------------------------------------------------------
# FUNCTIONS


def preds_to_dataarray(preds, model, valid_times, gsp_ids):
    """Put numpy array of predictions into a dataarray"""
    
    if model.use_quantile_regression:
        output_labels = model.output_quantiles
        output_labels = [f"forecast_mw_plevel_{int(q*100):02}" for q in model.output_quantiles]
        output_labels[output_labels.index("forecast_mw_plevel_50")] = "forecast_mw"
    else:
        output_labels = ["forecast_mw"]
        preds = preds[..., np.newaxis]

    da = xr.DataArray(
        data=preds,
        dims=["pv_system_id", "target_datetime_utc", "output_label"],
        coords=dict(
            pv_system_id=gsp_ids,
            target_datetime_utc=valid_times,
            output_label=output_labels,
        ),
    )
    return da

#TODO change this to load the PV sites data (metadata?) 
def get_sites_ds(config_path: str) -> xr.Dataset:
    """Load GSP data from the path in the data config.

    Args:
        config_path: Path to the data configuration file

    Returns:
        xarray.Dataset of PVLive truths and capacities
    """

    config = load_yaml_configuration(config_path)
    sme_site_datapipe = OpenPVFromNetCDFIterDataPipe(
            pv=config.input_data.pv
        )
    ds_sites = next(iter(sme_site_datapipe))

    return ds_sites


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

    # We are going to abuse the `create_t0_and_loc_datapipes()` function to find the init-times in
    # potential_init_times which we have input data for. To do this, we will feed in some fake GSP
    # data which has the potential_init_times as timestamps. This is a bit hacky but works for now

    # Set up init-times we would like to make predictions for
    potential_init_times = pd.date_range(start_datetime, end_datetime, freq=f"{FREQ_MINS}min")

    # We buffer the potential init-times so that we don't lose any init-times from the
    # start and end. Again this is a hacky step
    history_duration = pd.Timedelta(config.input_data.pv.history_minutes, "min")
    forecast_duration = pd.Timedelta(config.input_data.pv.forecast_minutes, "min")
    buffered_potential_init_times = pd.date_range(
        start_datetime - history_duration, end_datetime + forecast_duration, freq=f"{FREQ_MINS}min"
    )
    ds_fake_gsp = buffered_potential_init_times.to_frame().to_xarray().rename({"index": "time_utc"})
    ds_fake_gsp = ds_fake_gsp.rename({0: "gsp_pv_power_mw"})
    ds_fake_gsp = ds_fake_gsp.expand_dims("pv_system_id", axis=1)
    ds_fake_gsp = ds_fake_gsp.assign_coords(
        pv_system_id=[0],
        latitude=("pv_system_id", [0]),
        longitude=("pv_system_id", [0]),
    )
    ds_fake_gsp = ds_fake_gsp.gsp_pv_power_mw.astype(float) * 1e-18
    # Overwrite the GSP data which is already in the datapipes dict
    datapipes_dict["pv"] = IterableWrapper([ds_fake_gsp])

    # Use create_t0_and_loc_datapipes to get datapipe of init-times
    location_pipe, t0_datapipe = create_t0_and_loc_datapipes(
        datapipes_dict,
        configuration=config,
        key_for_t0="pv",
        shuffle=False,
    )

    # Create a full list of available init-times. Note that we need to loop over the t0s AND
    # locations to avoid the torch datapipes buffer overflow but we don't actually use the location
    available_init_times = [t0 for _, t0 in zip(location_pipe, t0_datapipe)]
    available_init_times = pd.to_datetime(available_init_times)

    logger.info(
        f"{len(available_init_times)} out of {len(potential_init_times)} "
        "requested init-times have required input data"
    )

    return available_init_times


def get_loctimes_datapipes(config_path):
    """Create location and init-time datapipes

    Args:
        config_path: Path to data config file

    Returns:
        tuple: A tuple of datapipes
            - Datapipe yielding locations
            - Datapipe yielding init-times
    """

    # Set up ID location query object
    ds_sites = get_sites_ds(config_path)
    gsp_id_to_loc = SiteLocationLookup(ds_sites.longitude, ds_sites.latitude)

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
    
    # Cycle the GSP locations
    location_pipe = IterableWrapper([[gsp_id_to_loc(gsp_id) for gsp_id in ALL_SITE_ML_IDS]]).repeat(
        num_t0s
    )

    # Shard and then unbatch the locations so that each worker will generate all samples for all
    # GSPs and for a single init-time
    location_pipe = location_pipe.sharding_filter()
    location_pipe = location_pipe.unbatch(unbatch_level=1) # might not need this part since the site datapipe is creating examples

    # Create times datapipe so each worker receives 317 copies of the same datetime for its batch
    t0_datapipe = IterableWrapper([[t0 for gsp_id in ALL_SITE_ML_IDS] for t0 in available_target_times])
    t0_datapipe = t0_datapipe.sharding_filter()
    t0_datapipe = t0_datapipe.unbatch(unbatch_level=1) # might not need this part since the site datapipe is creating examples


    t0_datapipe = t0_datapipe.set_length(num_t0s * len(ALL_SITE_ML_IDS))
    location_pipe = location_pipe.set_length(num_t0s * len(ALL_SITE_ML_IDS))

    return location_pipe, t0_datapipe


class ModelPipe:
    """A class to conveniently make and process predictions from batches"""

    def __init__(self, model, ds_gsp: xr.Dataset):
        """A class to conveniently make and process predictions from batches

        Args:
            model: PVNet GSP level model
            ds_gsp:xarray dataset of PVLive true values and capacities
        """
        self.model = model
        self.ds_gsp = ds_gsp

    def predict_batch(self, batch: NumpyBatch) -> xr.Dataset:
        """Run the batch through the model and compile the predictions into an xarray DataArray

        Args:
            batch: A batch of samples with inputs for each GSP for the same init-time

        Returns:
            xarray.Dataset of all GSP and national forecasts for the batch
        """
        # Unpack some variables from the batch
        id0 = batch[BatchKey.pv_t0_idx]
        
        t0 = batch[BatchKey.pv_time_utc].cpu().numpy().astype("datetime64[s]")[0, id0]
        n_valid_times = len(batch[BatchKey.pv_time_utc][0, id0 + 1 :])
        ds_gsp = self.ds_gsp
        model = self.model

        # Get valid times for this forecast
        valid_times = pd.to_datetime(
            [t0 + np.timedelta64((i + 1) * FREQ_MINS, "m") for i in range(n_valid_times)]
        )

        # Get effective capacities for this forecast
        gsp_capacities = ds_gsp.nominal_capacity_wp.values
        # Get the solar elevations. We need to un-normalise these from the values in the batch
        elevation = batch[BatchKey.pv_solar_elevation] * ELEVATION_STD + ELEVATION_MEAN
        # We only need elevation mask for forecasted values, not history
        elevation = elevation[:, id0 + 1 :]

        # Make mask dataset for sundown
        da_sundown_mask = xr.DataArray(
            data=elevation < MIN_DAY_ELEVATION,
            dims=["pv_system_id", "target_datetime_utc"],
            coords=dict(
                pv_system_id=ALL_SITE_ML_IDS,
                target_datetime_utc=valid_times,
            ),
        )

        with torch.no_grad():
            # Run batch through model to get 0-1 predictions for all GSPs
            device_batch = copy_batch_to_device(batch_to_tensor(batch), device)
            y_normed_gsp = model(device_batch).detach().cpu().numpy()
        da_normed_gsp = preds_to_dataarray(y_normed_gsp, model, valid_times, ALL_SITE_ML_IDS)

        # Multiply normalised forecasts by capacities and clip negatives
        # da_abs_gsp = da_normed_gsp.clip(0, None) * gsp_capacities[:, None, None]
        da_abs_gsp = da_normed_gsp.clip(0, None)
        # Apply sundown mask
        da_abs_gsp = da_abs_gsp.where(~da_sundown_mask).fillna(0.0)

        # Make national predictions using summation model
        
        # Concat the regional GSP and national predictions
        # da_abs_all = xr.concat([da_abs_national, da_abs_gsp], dim="gsp_id")
        # ds_abs_all = da_abs_all.to_dataset(name="hindcast")

        da_abs_gsp = da_abs_gsp.expand_dims(dim="init_time_utc", axis=0).assign_coords(
            init_time_utc=[t0]
        )

        return da_abs_gsp


def get_datapipe(config_path: str) -> NumpyBatch:
    """Construct datapipe yielding batches of concurrent samples for all GSPs

    Args:
        config_path: Path to the data configuration file

    Returns:
        NumpyBatch: Concurrent batch of samples for each GSP
    """

    # Construct location and init-time datapipes
    location_pipe, t0_datapipe = get_loctimes_datapipes(config_path)

    # Get the number of init-times
    # num_batches = len(t0_datapipe)
    num_batches = len(t0_datapipe) // len(ALL_SITE_ML_IDS)
    # Construct sample datapipes
    data_pipeline = construct_sliced_data_pipeline(
        config_path,
        location_pipe,
        t0_datapipe,
    )
    
    
    data_pipeline = DictDatasetIterDataPipe(
        {k: v for k, v in data_pipeline.items() if k != "config"},
    ).map(split_dataset_dict_dp)
    # .map(combine_to_single_dataset).map(split_dataset_dict_dp)

    # data_pipeline = IterableWrapper(data_pipeline)

    data_pipeline = data_pipeline.pvnet_site_convert_to_numpy_batch()

    # Batch so that each worker returns a batch of all locations for a single init-time
    # Also convert to tensor for model
    data_pipeline = (
        data_pipeline.batch(len(ALL_SITE_ML_IDS)).map(stack_np_examples_into_batch).map(batch_to_tensor)
    )
    # data_pipeline = data_pipeline.pvnet_site_convert_to_numpy_batch()
    data_pipeline = data_pipeline.set_length(num_batches)

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
    ds_gsp = get_sites_ds(config.datamodule.configuration)
    # Create a dataloader for the concurrent batches and use multiprocessing
    dataloader = DataLoader(batch_pipe, **dataloader_kwargs)
    # Load the PVNet model
    model, *_ = get_model_from_checkpoints([model_chckpoint_dir], val_best=True)
    model = model.eval().to(device)

    # Create object to make predictions for each input batch
    model_pipe = ModelPipe(model, ds_gsp)
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