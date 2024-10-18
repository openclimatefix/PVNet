"""
A script to run backtest for PVNet for specific sites

Use:

- This script uses hydra to construct the config, just like in `run.py`. So you need to make sure
  that the data config is set up appropriate for the model being run in this script
- The PVNet model checkpoint; the time range over which to make predictions are made;
  the site ids to produce forecasts for and the output directory where the results
  near the top of the script as hard coded user variables. These should be changed.

```
python scripts/backtest_sites.py
```

"""

try:
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass

import json
import logging
import os
import sys

import hydra
import numpy as np
import pandas as pd
import torch
import xarray as xr
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import CONFIG_NAME, PYTORCH_WEIGHTS_NAME
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
from ocf_datapipes.training.pvnet_site import (
    DictDatasetIterDataPipe,
    _get_datapipes_dict,
    construct_sliced_data_pipeline,
    split_dataset_dict_dp,
)
from ocf_datapipes.utils.consts import ELEVATION_MEAN, ELEVATION_STD
from omegaconf import DictConfig
from torch.utils.data import DataLoader, IterDataPipe, functional_datapipe
from torch.utils.data.datapipes.iter import IterableWrapper
from tqdm import tqdm

from pvnet.load_model import get_model_from_checkpoints
from pvnet.utils import SiteLocationLookup

# ------------------------------------------------------------------
# USER CONFIGURED VARIABLES TO RUN THE SCRIPT

# Directory path to save results
output_dir = "PLACEHOLDER"

# Local directory to load the PVNet checkpoint from. By default this should pull the best performing
# checkpoint on the val set
model_chckpoint_dir = "PLACEHOLDER"

hf_revision = None
hf_token = None
hf_model_id = None

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

# The frequency of the pv site data
FREQ_MINS = 30

# When sun as elevation below this, the forecast is set to zero
MIN_DAY_ELEVATION = 0

# Add all pv site ids here that you wish to produce forecasts for
ALL_SITE_IDS = []
# Need to be in ascending order
ALL_SITE_IDS.sort()

# ------------------------------------------------------------------
# FUNCTIONS


@functional_datapipe("pad_forward_pv")
class PadForwardPVIterDataPipe(IterDataPipe):
    """
    Pads forecast pv. 
    
    Sun position is calculated based off of pv time index
    and for t0's close to end of pv data can have wrong shape as pv starts
    to run out of data to slice for the forecast part.
    """

    def __init__(self, pv_dp: IterDataPipe, forecast_duration: np.timedelta64):
        """Init"""

        super().__init__()
        self.pv_dp = pv_dp
        self.forecast_duration = forecast_duration

    def __iter__(self):
        """Iter"""

        for xr_data in self.pv_dp:
            t0 = xr_data.time_utc.data[int(xr_data.attrs["t0_idx"])]
            pv_step = np.timedelta64(xr_data.attrs["sample_period_duration"])
            t_end = t0 + self.forecast_duration + pv_step
            time_idx = np.arange(xr_data.time_utc.data[0], t_end, pv_step)
            yield xr_data.reindex(time_utc=time_idx, fill_value=-1)


def load_model_from_hf(model_id: str, revision: str, token: str):

"""
Loads model from HuggingFace
"""
    model_file = hf_hub_download(
        repo_id=model_id,
        filename=PYTORCH_WEIGHTS_NAME,
        revision=revision,
        token=token,
    )

    # load config file
    config_file = hf_hub_download(
        repo_id=model_id,
        filename=CONFIG_NAME,
        revision=revision,
        token=token,
    )

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    model = hydra.utils.instantiate(config)

    state_dict = torch.load(model_file, map_location=torch.device("cuda"))
    model.load_state_dict(state_dict)  # type: ignore
    model.eval()  # type: ignore

    return model


def preds_to_dataarray(preds, model, valid_times, site_ids):
    """Put numpy array of predictions into a dataarray"""

    if model.use_quantile_regression:
        output_labels = [f"forecast_mw_plevel_{int(q*100):02}" for q in model.output_quantiles]
        output_labels[output_labels.index("forecast_mw_plevel_50")] = "forecast_mw"
    else:
        output_labels = ["forecast_mw"]
        preds = preds[..., np.newaxis]

    da = xr.DataArray(
        data=preds,
        dims=["pv_system_id", "target_datetime_utc", "output_label"],
        coords=dict(
            pv_system_id=site_ids,
            target_datetime_utc=valid_times,
            output_label=output_labels,
        ),
    )
    return da


# TODO change this to load the PV sites data (metadata?)
def get_sites_ds(config_path: str) -> xr.Dataset:
    """Load site data from the path in the data config.

    Args:
        config_path: Path to the data configuration file

    Returns:
        xarray.Dataset of PVLive truths and capacities
    """

    config = load_yaml_configuration(config_path)
    site_datapipe = OpenPVFromNetCDFIterDataPipe(pv=config.input_data.pv)
    ds_sites = next(iter(site_datapipe))

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
    # potential_init_times which we have input data for. To do this, we will feed in some fake site
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
    ds_fake_site = (
        buffered_potential_init_times.to_frame().to_xarray().rename({"index": "time_utc"})
    )
    ds_fake_site = ds_fake_site.rename({0: "site_pv_power_mw"})
    ds_fake_site = ds_fake_site.expand_dims("pv_system_id", axis=1)
    ds_fake_site = ds_fake_site.assign_coords(
        pv_system_id=[0],
        latitude=("pv_system_id", [0]),
        longitude=("pv_system_id", [0]),
    )
    ds_fake_site = ds_fake_site.site_pv_power_mw.astype(float) * 1e-18
    # Overwrite the site data which is already in the datapipes dict
    datapipes_dict["pv"] = IterableWrapper([ds_fake_site])

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
    site_id_to_loc = SiteLocationLookup(ds_sites.longitude, ds_sites.latitude)

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

    # Cycle the site locations
    location_pipe = IterableWrapper([[site_id_to_loc(site_id) for site_id in ALL_SITE_IDS]]).repeat(
        num_t0s
    )

    # Shard and then unbatch the locations so that each worker will generate all samples for all
    # sites and for a single init-time
    location_pipe = location_pipe.sharding_filter()
    location_pipe = location_pipe.unbatch(
        unbatch_level=1
    )  # might not need this part since the site datapipe is creating examples

    # Create times datapipe so each worker receives
    # len(ALL_SITE_IDS) copies of the same datetime for its batch
    t0_datapipe = IterableWrapper(
        [[t0 for site_id in ALL_SITE_IDS] for t0 in available_target_times]
    )
    t0_datapipe = t0_datapipe.sharding_filter()
    t0_datapipe = t0_datapipe.unbatch(
        unbatch_level=1
    )  # might not need this part since the site datapipe is creating examples

    t0_datapipe = t0_datapipe.set_length(num_t0s * len(ALL_SITE_IDS))
    location_pipe = location_pipe.set_length(num_t0s * len(ALL_SITE_IDS))

    return location_pipe, t0_datapipe


class ModelPipe:
    """A class to conveniently make and process predictions from batches"""

    def __init__(self, model, ds_site: xr.Dataset):
        """A class to conveniently make and process predictions from batches

        Args:
            model: PVNet site level model
            ds_site:xarray dataset of pv site true values and capacities
        """
        self.model = model
        self.ds_site = ds_site

    def predict_batch(self, batch: NumpyBatch) -> xr.Dataset:
        """Run the batch through the model and compile the predictions into an xarray DataArray

        Args:
            batch: A batch of samples with inputs for each site for the same init-time

        Returns:
            xarray.Dataset of all site and national forecasts for the batch
        """
        # Unpack some variables from the batch
        id0 = batch[BatchKey.pv_t0_idx]

        t0 = batch[BatchKey.pv_time_utc].cpu().numpy().astype("datetime64[s]")[0, id0]
        n_valid_times = len(batch[BatchKey.pv_time_utc][0, id0 + 1 :])
        model = self.model

        # Get valid times for this forecast
        valid_times = pd.to_datetime(
            [t0 + np.timedelta64((i + 1) * FREQ_MINS, "m") for i in range(n_valid_times)]
        )

        # Get effective capacities for this forecast
        site_capacities = self.ds_site.nominal_capacity_wp.values
        # Get the solar elevations. We need to un-normalise these from the values in the batch
        elevation = batch[BatchKey.pv_solar_elevation] * ELEVATION_STD + ELEVATION_MEAN
        # We only need elevation mask for forecasted values, not history
        elevation = elevation[:, id0 + 1 :]

        # Make mask dataset for sundown
        da_sundown_mask = xr.DataArray(
            data=elevation < MIN_DAY_ELEVATION,
            dims=["pv_system_id", "target_datetime_utc"],
            coords=dict(
                pv_system_id=ALL_SITE_IDS,
                target_datetime_utc=valid_times,
            ),
        )

        with torch.no_grad():
            # Run batch through model to get 0-1 predictions for all sites
            device_batch = copy_batch_to_device(batch_to_tensor(batch), device)
            y_normed_site = model(device_batch).detach().cpu().numpy()
        da_normed_site = preds_to_dataarray(y_normed_site, model, valid_times, ALL_SITE_IDS)

        # Multiply normalised forecasts by capacities and clip negatives
        da_abs_site = da_normed_site.clip(0, None) * site_capacities[:, None, None]

        # Apply sundown mask
        da_abs_site = da_abs_site.where(~da_sundown_mask).fillna(0.0)

        da_abs_site = da_abs_site.expand_dims(dim="init_time_utc", axis=0).assign_coords(
            init_time_utc=np.array([t0], dtype="datetime64[ns]")
        )

        return da_abs_site


def get_datapipe(config_path: str) -> NumpyBatch:
    """Construct datapipe yielding batches of concurrent samples for all sites

    Args:
        config_path: Path to the data configuration file

    Returns:
        NumpyBatch: Concurrent batch of samples for each site
    """

    # Construct location and init-time datapipes
    location_pipe, t0_datapipe = get_loctimes_datapipes(config_path)

    # Get the number of init-times
    # num_batches = len(t0_datapipe)
    num_batches = len(t0_datapipe) // len(ALL_SITE_IDS)
    # Construct sample datapipes
    data_pipeline = construct_sliced_data_pipeline(
        config_path,
        location_pipe,
        t0_datapipe,
    )

    config = load_yaml_configuration(config_path)
    data_pipeline["pv"] = data_pipeline["pv"].pad_forward_pv(
        forecast_duration=np.timedelta64(config.input_data.pv.forecast_minutes, "m")
    )

    data_pipeline = DictDatasetIterDataPipe(
        {k: v for k, v in data_pipeline.items() if k != "config"},
    ).map(split_dataset_dict_dp)

    data_pipeline = data_pipeline.pvnet_site_convert_to_numpy_batch()

    # Batch so that each worker returns a batch of all locations for a single init-time
    # Also convert to tensor for model
    data_pipeline = (
        data_pipeline.batch(len(ALL_SITE_IDS))
        .map(stack_np_examples_into_batch)
        .map(batch_to_tensor)
    )
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
    # Each batch includes a sample for each of the n sites for a single init-time
    batch_pipe = get_datapipe(config.datamodule.configuration)
    num_batches = len(batch_pipe)
    # Load the site data as an xarray object
    ds_site = get_sites_ds(config.datamodule.configuration)
    # Create a dataloader for the concurrent batches and use multiprocessing
    dataloader = DataLoader(batch_pipe, **dataloader_kwargs)
    # Load the PVNet model
    if model_chckpoint_dir:
        model, *_ = get_model_from_checkpoints([model_chckpoint_dir], val_best=True)
    elif hf_model_id:
        model = load_model_from_hf(hf_model_id, hf_revision, hf_token)
    else:
        raise ValueError("Provide a model checkpoint or a HuggingFace model")

    model = model.eval().to(device)

    # Create object to make predictions for each input batch
    model_pipe = ModelPipe(model, ds_site)
    # Loop through the batches
    pbar = tqdm(total=num_batches)
    for i, batch in zip(range(num_batches), dataloader):
        try:
            # Make predictions for the init-time
            ds_abs_all = model_pipe.predict_batch(batch)

            t0 = ds_abs_all.init_time_utc.values[0]

            # Save the predictions
            filename = f"{output_dir}/{t0}.nc"
            ds_abs_all.to_netcdf(filename)

            pbar.update()
        except Exception as e:
            print(f"Exception {e} at batch {i}")
            pass

    # Close down
    pbar.close()
    del dataloader


if __name__ == "__main__":
    main()
