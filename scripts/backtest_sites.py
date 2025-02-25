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
from ocf_data_sampler.torch_datasets.datasets.site import SitesDataset
from ocf_data_sampler.config import load_yaml_configuration
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from pvnet.load_model import get_model_from_checkpoints

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
start_datetime = "2022-05-07 13:00"
end_datetime = "2022-05-07 16:30"

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
        xarray.Dataset of PV sites data
    """
    dataset = SitesDataset(config_path)
    return dataset.datasets_dict["site"]



class ModelPipe:
    """A class to conveniently make and process predictions from batches"""

    def __init__(self, model, ds_site: xr.Dataset, config_path: str):
        """A class to conveniently make and process predictions from batches

        Args:
            model: PVNet site level model
            ds_site: xarray dataset of pv site true values and capacities
        """
        self.model = model
        self.ds_site = ds_site
        self.config_path = config_path

    def predict_batch(self, sample: dict) -> xr.Dataset:
        """Run the sample through the model and compile the predictions into an xarray DataArray

        Args:
            sample: A sample containing inputs for a site

        Returns:
            xarray.Dataset of site forecasts for the sample
        """
        # Convert sample to tensor and move to device
        sample_tensor = {k: torch.from_numpy(v).to(device) for k, v in sample.items()}

        config = load_yaml_configuration(self.config_path)

        interval_start = np.timedelta64(config.input_data.site.interval_start_minutes, "m")
        interval_end = np.timedelta64(config.input_data.site.interval_end_minutes, "m")
        time_resolution = np.timedelta64(config.input_data.site.time_resolution_minutes, "m")

        t0 = pd.Timestamp(sample["site_init_time_utc"][0])
        site_id = sample["site_id"][0]

        #Get valid times for this forecast
        valid_times = pd.date_range(
            start=t0 + pd.Timedelta(interval_start),
            end=t0 + pd.Timedelta(interval_end),
            freq=f"{time_resolution.astype(int)}min"
        )

        # Get capacity for this site
        site_capacity = float(self.ds_site.sel(site_id=site_id).capacity_kwp)

        # Get solar elevation and create sundown mask
        elevation = sample["site_solar_elevation"]
        da_sundown_mask = xr.DataArray(
            data=elevation < MIN_DAY_ELEVATION,
            dims=["target_datetime_utc"],
            coords=dict(target_datetime_utc=valid_times),
        )

        with torch.no_grad():
            # Run through model to get 0-1 predictions
            y_normed = self.model(sample_tensor).detach().cpu().numpy()

        da_normed = preds_to_dataarray(y_normed, self.model, valid_times, [site_id])

        # Multiply normalised forecasts by capacity and clip negatives
        da_abs = da_normed.clip(0, None) * site_capacity

        # Apply sundown mask
        da_abs = da_abs.where(~da_sundown_mask).fillna(0.0)

        da_abs = da_abs.expand_dims(dim="init_time_utc", axis=0).assign_coords(
            init_time_utc=np.array([t0], dtype="datetime64[ns]")
        )

        return da_abs

def get_datapipe(config_path: str):
    """Construct dataset for all sites

    Args:
        config_path: Path to the data configuration file

    Returns:
        SitesDataset: Dataset containing samples for each site
    """
    # Create dataset with time range filter
    dataset = SitesDataset(
        config_path,
        start_time=start_datetime,
        end_time=end_datetime
    )

    # Filter for specific site IDs
    dataset.valid_t0_and_site_ids = dataset.valid_t0_and_site_ids[
        dataset.valid_t0_and_site_ids.site_id.isin(ALL_SITE_IDS)
    ]

    return dataset


@hydra.main(config_path="../configs", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig):
    """Runs the backtest"""

    dataloader_kwargs = dict(
        shuffle=False,
        batch_size=None,
        num_workers=config.datamodule.num_workers,
        pin_memory=False,
        drop_last=False,
        prefetch_factor=config.datamodule.prefetch_factor,
        persistent_workers=False,
    )

    # Set up output dir
    os.makedirs(output_dir)

    # Create dataset
    dataset = get_datapipe(config.datamodule.configuration)

    # Load the site data
    ds_site = get_sites_ds(config.datamodule.configuration)

    # Create a dataloader
    dataloader = DataLoader(dataset, **dataloader_kwargs)

    # Load the PVNet model
    if model_chckpoint_dir:
        model, *_ = get_model_from_checkpoints([model_chckpoint_dir], val_best=True)
    elif hf_model_id:
        model = load_model_from_hf(hf_model_id, hf_revision, hf_token)
    else:
        raise ValueError("Provide a model checkpoint or a HuggingFace model")

    model = model.eval().to(device)

    # Create object to make predictions
    model_pipe = ModelPipe(model, ds_site, config.datamodule.configuration)

    # Loop through the samples
    pbar = tqdm(total=len(dataset))
    for i, sample in enumerate(dataloader):
        try:
            # Make predictions
            ds_abs_all = model_pipe.predict_batch(sample)

            t0 = ds_abs_all.init_time_utc.values[0]

            # Save the predictions
            filename = f"{output_dir}/{t0}.nc"
            ds_abs_all.to_netcdf(filename)

            pbar.update()
        except Exception as e:
            print(f"Exception {e} at sample {i}")
            pass

    pbar.close()
    del dataloader

if __name__ == "__main__":
    main()
