try:
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass

import os, sys
from omegaconf import DictConfig
from tqdm import tqdm
import json
import tempfile
import yaml

import hydra
from torch.utils.data import DataLoader, Dataset
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import CONFIG_NAME, PYTORCH_WEIGHTS_NAME

import numpy as np
import pandas as pd
import xarray as xr

from pvnet.load_model import get_model_from_checkpoints
from pvnet.models.multimodal.multimodal import Model

from ocf_datapipes.batch import BatchKey, NumpyBatch, stack_np_examples_into_batch, batch_to_tensor, copy_batch_to_device
from ocf_datapipes.config.load import load_yaml_configuration
from ocf_datapipes.utils.consts import ELEVATION_MEAN, ELEVATION_STD 

from ocf_data_sampler.torch_datasets.pvnet_uk_regional import PVNetUKRegionalDataset

import logging

log = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


#_________VARIABLES_________

output_dir = "/home/alex/wp6/PVNet/backtest" # directory where to store predictions

# Time window for which to run the backtest
start_time = "2024-12-31"
end_time = "2025-01-05"

# The frequency at which to create forecast horizons
FREQ_MINS = 30

# When sun at elevation below this, the forecast is set to zero
MIN_DAY_ELEVATION = 0

# All regional GSP IDs
ALL_GSP_IDS = np.arange(1, 318)

# Whether to include all GSP IDs in the batch. If false defaults to the batch size in datamodule
predict_all_gsps = False 

#_________MODEL VARIABLES_________

hf_model_id = "openclimatefix/pvnet_uk_region"
hf_revision = "0bc344fafb2232fb0b6bb0bf419f0449fe11c643"
hf_token = None

#_________DATA_PATHS_________

# paths to populate the config with after loading from HF 
ecmwf_path = "/home/alex/wp6/data/2025010100.zarr"
gsp_path = "/home/alex/wp6/data/pv_gsp.zarr"

# member if using ECMWF ensambles. Set to None to ignore
ensemble_member = 1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def adapt_data_config(config: dict) -> dict:
    """Populates data paths with ones set in variable section of the script and adds
    ensemble member if provided

    Args:
        config: dict = Config read off of HuggingFace

    Returns:
        adapted config
    """


    if ecmwf_path and "ecmwf" in config["input_data"]["nwp"]:
        config["input_data"]["nwp"]["ecmwf"]["nwp_zarr_path"] = ecmwf_path
        log.info(f"Updated ecmwf path to {ecmwf_path}")
    if gsp_path and "gsp" in config["input_data"]:
        config["input_data"]["gsp"]["gsp_zarr_path"] = gsp_path
        log.info(f"Updated gsp path to {gsp_path}")

    if ensemble_member:
        config["input_data"]["nwp"]["ecmwf"]["nwp_ensemble_member"] = ensemble_member
        log.info(f"Updated ECMWF ensemble member to {ensemble_member}")

    return config


def load_model_from_hf(model_id: str, revision: str, token: str) -> tuple[Model, str]:
    """Loads model and data config from HuggingFace
    Adapts and saves data config to be used by datasampler. 

    Args:
        model_id: model repo on HF (eg 'openclimatefix/model')
        revision: revision tag for model repo
        token: access token. Needed for access to private repos

    Returns:
        model on device and path to adapted data config
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
    model.load_state_dict(state_dict)
    model.eval().to(device)

    # load data config
    data_config_file = hf_hub_download(
        repo_id=model_id,
        filename="data_config.yaml",
        revision=revision,
        token=token,
    )

    with open(data_config_file, "r", encoding="utf-8") as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)

    data_config_path = f"{output_dir}/data_config.yaml"

    with open(data_config_path, "w") as file:
        yaml.dump(adapt_data_config(data_config), file, default_flow_style=False)

    return model, data_config_path


def get_gsp_capacities(config_path: str) -> xr.DataArray:
    """Load GSP data from the path in the data config.

    Args:
        config_path: Path to the data configuration file

    Returns:
        xarray.DataArray of PVLive capacities
    """

    config = load_yaml_configuration(config_path)

    # Load GSP generation xr.Dataset
    ds = xr.open_zarr(config.input_data.gsp.gsp_zarr_path)

    # Rename to standard time name
    ds = ds.rename({"datetime_gmt": "time_utc"})

    return ds.capacity_mwp


def copy_sample_to_device(sample: dict, device: torch.device) -> dict:
    """
    Moves tensor leaves in a nested dict to a new device

    Args:
        sample: nested dict with tensors to move
        device: Device to move tensors to

    Returns:
        A dict with tensors moved to new device
    """
    sample_copy = {}

    for k, v in sample.items():
        if isinstance(v, dict):
            # Recursion to reach the nested NWP
            sample_copy[k] = copy_sample_to_device(v, device)
        elif isinstance(v, torch.Tensor):
            sample_copy[k] = v.to(device)
        else:
            sample_copy[k] = v
    return sample_copy


def preds_to_dataarray(
    preds: np.ndarray, 
    model: Model, 
    valid_times: pd.DatetimeIndex, 
    gsp_ids: np.ndarray) -> xr.DataArray:
    """Put numpy array of predictions into a dataarray
    
    Args:
        preds: batch predictions on cpu in numpy
        model: model
        valid_times: index of forecast horizons
        gsp_ids: list of GSP IDs present in the batch
    
    Returns:
        xarray DataArray of predictions
    """

    if model.use_quantile_regression:
        output_labels = [f"forecast_mw_plevel_{int(q*100):02}" for q in model.output_quantiles]
        output_labels[output_labels.index("forecast_mw_plevel_50")] = "forecast_mw"
    else:
        output_labels = ["forecast_mw"]
        preds = preds[..., np.newaxis]

    da = xr.DataArray(
        data=preds,
        dims=["gsp_id", "target_datetime_utc", "output_label"],
        coords=dict(
            gsp_id=gsp_ids,
            target_datetime_utc=valid_times,
            output_label=output_labels,
        ),
    )
    return da


class ModelPipe:
    """A class to conveniently make and process predictions from batches"""

    def __init__(self, model: Model, gsp_capacities: xr.DataArray):
        """A class to conveniently make and process predictions from batches

        Args:
            model: PVNet GSP level model
            gsp_capacities: xarray dataarray of PVLive capacities
        """
        self.model = model
        self.gsp_capacities = gsp_capacities

    def predict_sample(self, batch: NumpyBatch) -> xr.Dataset:
        """Run the batch through the model and compile the predictions into an xarray DataArray

        Args:
            batch: A batch of samples with inputs for each GSP for the same init-time

        Returns:
            xarray.Dataset of all GSP forecasts for the batch
        """

        log.debug("Predicting batch")

        # Unpack some variables from the batch
        id0 = int(batch[BatchKey.gsp_t0_idx])
        t0 = batch[BatchKey.gsp_time_utc].astype("datetime64[ns]")[0, id0]
        n_valid_times = len(batch[BatchKey.gsp_time_utc][0, id0 + 1 :])
        gsp_capacities = self.gsp_capacities
        model = self.model

        log.debug(f"Found t0 {t0}. Will forecast for {n_valid_times} points at frequency {FREQ_MINS} minutes")

        # Get valid times for this forecast
        valid_times = pd.to_datetime(
            [t0 + np.timedelta64((i + 1) * FREQ_MINS, "m") for i in range(n_valid_times)]
        )

        log.debug(f"Valid times to forecast for: {valid_times}")

        # Get effective capacities for this forecast
        gsp_capacities = gsp_capacities.sel(
            time_utc=t0, gsp_id=batch[BatchKey.gsp_id]
        ).values

        log.debug(f"GSP capacities identified. Shape: {gsp_capacities.shape}")

        # Get the solar elevations. We need to un-normalise these from the values in the batch
        # The new dataloader normalises the data to [0, 1]
        elevation = (batch[BatchKey.gsp_solar_elevation] - 0.5) * 180

        log.debug(f"Denormalised solar elevation: {elevation}")

        # We only need elevation mask for forecasted values, not history
        elevation = elevation[:, id0 + 1 :]

        # Make mask dataset for sundown
        da_sundown_mask = xr.DataArray(
            data=elevation < MIN_DAY_ELEVATION,
            dims=["gsp_id", "target_datetime_utc"],
            coords=dict(
                gsp_id=batch[BatchKey.gsp_id],
                target_datetime_utc=valid_times,
            ),
        )

        log.debug(f"Sundown mask created. Filtering to solar elevation less than {MIN_DAY_ELEVATION}")
        log.debug(f"Sundown mask is {da_sundown_mask}")

        with torch.no_grad():
            log.debug("Running batch through model...")
            # Run batch through model to get 0-1 predictions for all GSPs
            device_batch = copy_batch_to_device(batch_to_tensor(batch), device)
            log.debug(f"Moved batch to device: {device}")
            y_normed_gsp = model(device_batch).detach().cpu().numpy()

        log.debug(f"Prediction complete!")
        log.debug(f"Prediction shape: {y_normed_gsp.shape}")

        log.debug("Converting to dataarray...")

        da_normed_gsp = preds_to_dataarray(
            preds=y_normed_gsp, 
            model=model, 
            valid_times=valid_times, 
            gsp_ids=batch[BatchKey.gsp_id].numpy())

        # Multiply normalised forecasts by capacities and clip negatives
        log.debug("Denormalising predictions...")

        da_abs_gsp = da_normed_gsp.clip(0, None) * gsp_capacities[:, None, None]

        # Apply sundown mask
        log.debug("Upplying sundown mask...")

        da_abs_gsp = da_abs_gsp.where(~da_sundown_mask).fillna(0.0)
        da_abs_gsp = da_abs_gsp.expand_dims(dim="init_time_utc", axis=0).assign_coords(
            init_time_utc=[t0]
        )

        log.debug("Prediction ready to save!")

        return da_abs_gsp



@hydra.main(config_path="../configs/", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig) -> None:
    """ 
    Opens model, constructs dataloader, feeds created samples through the model 
    and saves predictions
    """

    log.info(
        f"Backtest initiated. Selected time window: {start_time} to {end_time}. "
        f"Total of {len(ALL_GSP_IDS)} GSP IDs requested"
    )
    log.info(f"Device used: {device}")

    # Set up output dir
    os.makedirs(output_dir)

    log.info(f"Output directory created: {output_dir}")
    
    # Load data from HuggingFace
    model, data_config_path = load_model_from_hf(hf_model_id, hf_revision, hf_token)
    log.info(f"Model loaded from HuggingFace: {hf_model_id}/{hf_revision}")

    # If all-gsp batch requested, override batch size
    if predict_all_gsps:
        batch_size = len(ALL_GSP_IDS)
        log.warning(f"All GSP IDs are used per batch. Batch size increased to {batch_size}")
    else:
        batch_size = config.datamodule.batch_size

    # Prepare DataLoader
    dataloader_kwargs = dict(
        shuffle=False, # Go through samples t0-first (needs the swap in ocf-data-sampler pvnet_uk_regional L525)
        batch_size=batch_size,
        sampler=None,
        batch_sampler=None,
        num_workers=config.datamodule.num_workers,
        collate_fn=stack_np_examples_into_batch,
        pin_memory=False,  # Only using CPU to prepare samples so pinning is not beneficial
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        prefetch_factor=config.datamodule.prefetch_factor,
        persistent_workers=False,  # Not needed since we only enter the dataloader loop once
    )


    # Get the dataset   
    dataset = PVNetUKRegionalDataset(
        config_filename=data_config_path, 
        start_time=start_time, 
        end_time=end_time,
        gsp_ids=ALL_GSP_IDS,
        )

    num_samples = dataset.__len__()

    # Create a dataloader for the concurrent samples and use multiprocessing
    dataloader = DataLoader(dataset, **dataloader_kwargs)

    log.info(f"Dataloader created with batch size {config.datamodule.batch_size}, " 
    f"num_workers {config.datamodule.num_workers}, prefetch factor {config.datamodule.prefetch_factor}")

    # Load the GSP capacities as an xarray object
    gsp_capacities = get_gsp_capacities(config.datamodule.configuration)

    # Create object to make predictions for each input sample
    model_pipe = ModelPipe(model=model, gsp_capacities=gsp_capacities)
    log.info("Model Pipe initialised. Starting inference...")

    # Loop through the samplees
    pbar = tqdm(total=num_samples//batch_size)
    for i, sample in zip(range(num_samples), dataloader):

        # Make predictions for the init-time
        da_abs_gsp = model_pipe.predict_sample(sample)

        # Save the predictions
        t0 = da_abs_gsp.init_time_utc.values[0]

        filename = f"{t0.astype('datetime64[m]')}_ID_{'_'.join(sample[BatchKey.gsp_id].numpy().astype(str))}.nc"

        da_abs_gsp.to_netcdf(f"{output_dir}/{filename}")

        pbar.update()

    # Close down
    pbar.close()
    del dataloader


if __name__ == "__main__":
    main()
