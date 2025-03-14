import satellite_patch
import fix_optimizer

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

from ocf_data_sampler.config.load import load_yaml_configuration
from ocf_data_sampler.torch_datasets.datasets.pvnet_uk import PVNetUKRegionalDataset
from ocf_data_sampler.numpy_sample.gsp import GSPSampleKey
from ocf_data_sampler.numpy_sample.collate import stack_np_samples_into_batch
from ocf_data_sampler.sample.base import NumpyBatch, batch_to_tensor, copy_batch_to_device

import logging


log = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


#_________VARIABLES_________

output_dir = "/mnt/felix-output-real/backtest_intra"
start_time = "2023-01-07"
end_time = "2023-12-30"
FREQ_MINS = 30
MIN_DAY_ELEVATION = 0
ALL_GSP_IDS = np.arange(1, 318)

#_________MODEL VARIABLES_________

# Intraday - ECMWF ONLY
hf_model_id = "openclimatefix/pvnet_uk_region"
hf_revision = "d81a9cf8adca49739ea6a3d031e36510f44744a1"
hf_token = None

#_________DATA_PATHS_________

# paths to populate the config with after loading from HF 
gsp_path = "/mnt/uk-all-inputs-v3/pv_gsp/pvlive_gsp.zarr"

ecmwf_paths = [
    "/mnt/uk-all-inputs-v3/nwp/ecmwf/UK_v3/ECMWF_2019.zarr",
    "/mnt/uk-all-inputs-v3/nwp/ecmwf/UK_v3/ECMWF_2020.zarr",
    "/mnt/uk-all-inputs-v3/nwp/ecmwf/UK_v3/ECMWF_2021.zarr",
    "/mnt/uk-all-inputs-v3/nwp/ecmwf/UK_v3/ECMWF_2022.zarr",
    "/mnt/uk-all-inputs-v3/nwp/ecmwf/UK_v3/ECMWF_2023.zarr",
    "/mnt/uk-all-inputs-v3/nwp/ecmwf/UK_v3/ECMWF_2024.zarr"
]

ukv_paths = [
    "/mnt/uk-all-inputs-v3/nwp/ukv/UKV_v8/UKV_2023.zarr",
    # "/mnt/uk-all-inputs-v3/nwp/ukv/UKV_v8/UKV_2024.zarr",
]

satellite_paths = [
    "/mnt/uk-all-inputs-v3/sat/v3/2023-11_nonhrv.zarr",
]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def adapt_data_config(config: dict) -> dict:
    """Populates data paths with ones set in variable section of the script and adds
    ensemble member if provided

    Args:
        config: dict = Config read off of HuggingFace

    Returns:
        adapted config
    """

    if "ecmwf" in config["input_data"]["nwp"]:
        config["input_data"]["nwp"]["ecmwf"]["nwp_zarr_path"] = ecmwf_paths
        log.info(f"Updated ecmwf path to include multiple zarr files: {ecmwf_paths}")
    if "ukv" in config["input_data"]["nwp"]:
        config["input_data"]["nwp"]["ukv"]["nwp_zarr_path"] = ukv_paths
        log.info(f"Updated ukv path to include multiple zarr files: {ukv_paths}")
    if "satellite" in config["input_data"]:
        config["input_data"]["satellite"]["satellite_zarr_path"] = satellite_paths
        log.info(f"Updated satellite path to include multiple zarr files: {satellite_paths}")
    if "gsp" in config["input_data"]:
        config["input_data"]["gsp"]["gsp_zarr_path"] = gsp_path
        log.info(f"Updated gsp path to {gsp_path}")

    return config


def adapt_config_for_schema(config):
    """
    Adapt the configuration to match the expected schema.
    This function translates between the field names in our config and what ocf_data_sampler expects.
    """
    import copy
    adapted_config = copy.deepcopy(config)
    
    # ECMWF configuration
    if "ecmwf" in adapted_config["input_data"]["nwp"]:
        ecmwf_config = adapted_config["input_data"]["nwp"]["ecmwf"]
        if "nwp_zarr_path" in ecmwf_config:
            ecmwf_config["zarr_path"] = ecmwf_config.pop("nwp_zarr_path")
        if "nwp_channels" in ecmwf_config:
            channels = ecmwf_config.pop("nwp_channels")
            if "sde" in channels:
                channels[channels.index("sde")] = "sd"            
            ecmwf_config["channels"] = channels
        if "nwp_provider" in ecmwf_config:
            ecmwf_config["provider"] = ecmwf_config.pop("nwp_provider")
        if "nwp_image_size_pixels_height" in ecmwf_config:
            ecmwf_config["image_size_pixels_height"] = ecmwf_config.pop("nwp_image_size_pixels_height")
        if "nwp_image_size_pixels_width" in ecmwf_config:
            ecmwf_config["image_size_pixels_width"] = ecmwf_config.pop("nwp_image_size_pixels_width")        
        if "interval_start_minutes" not in ecmwf_config:
            ecmwf_config["interval_start_minutes"] = -120
        if "interval_end_minutes" not in ecmwf_config:
            ecmwf_config["interval_end_minutes"] = 480
        if "max_staleness_minutes" not in ecmwf_config:
            ecmwf_config["max_staleness_minutes"] = None
        if "forecast_minutes" in ecmwf_config:
            del ecmwf_config["forecast_minutes"]
        if "history_minutes" in ecmwf_config:
            del ecmwf_config["history_minutes"]

    # UKV configuration
    if "ukv" in adapted_config["input_data"]["nwp"]:
        ukv_config = adapted_config["input_data"]["nwp"]["ukv"]
        if "nwp_zarr_path" in ukv_config:
            ukv_config["zarr_path"] = ukv_config.pop("nwp_zarr_path")
        if "nwp_channels" in ukv_config:          
            ukv_config["channels"] = ukv_config.pop("nwp_channels")
        if "nwp_provider" in ukv_config:
            ukv_config["provider"] = ukv_config.pop("nwp_provider")
        if "nwp_image_size_pixels_height" in ukv_config:
            ukv_config["image_size_pixels_height"] = ukv_config.pop("nwp_image_size_pixels_height")
        if "nwp_image_size_pixels_width" in ukv_config:
            ukv_config["image_size_pixels_width"] = ukv_config.pop("nwp_image_size_pixels_width")        
        if "interval_start_minutes" not in ukv_config:
            ukv_config["interval_start_minutes"] = -120
        if "interval_end_minutes" not in ukv_config:
            ukv_config["interval_end_minutes"] = 480

        ukv_config["time_resolution_minutes"] = 60

        if "max_staleness_minutes" in ukv_config:
            ukv_config["max_staleness_minutes"] = 1800
        if "max_staleness_minutes" not in ukv_config:
            ukv_config["max_staleness_minutes"] = 1800

        if "forecast_minutes" in ukv_config:
            del ukv_config["forecast_minutes"]
        if "history_minutes" in ukv_config:
            del ukv_config["history_minutes"]

    # Satellite configuration
    if "satellite" in adapted_config["input_data"]:
        satellite_config = adapted_config["input_data"]["satellite"]   
        if "satellite_zarr_path" in satellite_config:
            satellite_config["zarr_path"] = satellite_config.pop("satellite_zarr_path")
        if "satellite_channels" in satellite_config:
            satellite_config["channels"] = satellite_config.pop("satellite_channels")
        if "satellite_image_size_pixels_height" in satellite_config:
            satellite_config["image_size_pixels_height"] = satellite_config.pop("satellite_image_size_pixels_height")
        if "satellite_image_size_pixels_width" in satellite_config:
            satellite_config["image_size_pixels_width"] = satellite_config.pop("satellite_image_size_pixels_width")
        if "interval_start_minutes" not in satellite_config:
            satellite_config["interval_start_minutes"] = -90
        if "interval_end_minutes" not in satellite_config:
            satellite_config["interval_end_minutes"] = 0  
        if "forecast_minutes" in satellite_config:
            del satellite_config["forecast_minutes"]
        if "history_minutes" in satellite_config:
            del satellite_config["history_minutes"]
        if "dropout_timedeltas_minutes" in satellite_config and satellite_config["dropout_timedeltas_minutes"] is None:
            satellite_config["dropout_timedeltas_minutes"] = []
        elif "dropout_timedeltas_minutes" not in satellite_config:
            satellite_config["dropout_timedeltas_minutes"] = []

        if "live_delay_minutes" in satellite_config:
            del satellite_config["live_delay_minutes"]

    # GSP configuration
    if "gsp" in adapted_config["input_data"]:
        gsp_config = adapted_config["input_data"]["gsp"]        
        if "gsp_zarr_path" in gsp_config:
            gsp_config["zarr_path"] = gsp_config.pop("gsp_zarr_path")        
        if "interval_start_minutes" not in gsp_config:
            gsp_config["interval_start_minutes"] = -120
        if "interval_end_minutes" not in gsp_config:
            gsp_config["interval_end_minutes"] = 480
        if "dropout_timedeltas_minutes" in gsp_config and gsp_config["dropout_timedeltas_minutes"] is None:
            gsp_config["dropout_timedeltas_minutes"] = []
        elif "dropout_timedeltas_minutes" not in gsp_config:
            gsp_config["dropout_timedeltas_minutes"] = []
        if "forecast_minutes" in gsp_config:
            del gsp_config["forecast_minutes"]
        if "history_minutes" in gsp_config:
            del gsp_config["history_minutes"]
    
    # Uncomment below for current DA model
    # adapted_config["input_data"]["satellite"] = None
    return adapted_config


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

    config_file = hf_hub_download(
        repo_id=model_id,
        filename=CONFIG_NAME,
        revision=revision,
        token=token,
    )

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    model = hydra.utils.instantiate(config)

    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict)
    model.eval().to(device)

    data_config_file = hf_hub_download(
        repo_id=model_id,
        filename="data_config.yaml",
        revision=revision,
        token=token,
    )

    with open(data_config_file, "r", encoding="utf-8") as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)

    # Adapt config and correlate schema
    adapted_config = adapt_data_config(data_config)    
    schema_adapted_config = adapt_config_for_schema(adapted_config)
    data_config_path = f"{output_dir}/data_config.yaml"

    with open(data_config_path, "w") as file:
        yaml.dump(schema_adapted_config, file, default_flow_style=False)

    return model, data_config_path


def get_gsp_capacities(config_path: str) -> xr.DataArray:
    """Load GSP data from the path in the data config.

    Args:
        config_path: Path to the data configuration file

    Returns:
        xarray.DataArray of PVLive capacities
    """
    with open(config_path, 'r') as file:
        raw_config = yaml.safe_load(file)
    
    gsp_zarr_path = raw_config['input_data']['gsp']['zarr_path']
    ds = xr.open_zarr(gsp_zarr_path)
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
        id0 = int(batch[GSPSampleKey.t0_idx])
        t0 = batch[GSPSampleKey.time_utc].astype("datetime64[ns]")[0, id0]
        n_valid_times = len(batch[GSPSampleKey.time_utc][0, id0 + 1 :])
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
            time_utc=t0, gsp_id=batch[GSPSampleKey.gsp_id]
        ).values

        log.debug(f"GSP capacities identified. Shape: {gsp_capacities.shape}")

        # Get the solar elevations. We need to un-normalise these from the values in the batch
        # The new dataloader normalises the data to [0, 1]
        elevation = (batch[GSPSampleKey.solar_elevation] - 0.5) * 180

        log.debug(f"Denormalised solar elevation: {elevation}")

        # We only need elevation mask for forecasted values, not history
        elevation = elevation[:, id0 + 1 :]

        # Make mask dataset for sundown
        da_sundown_mask = xr.DataArray(
            data=elevation < MIN_DAY_ELEVATION,
            dims=["gsp_id", "target_datetime_utc"],
            coords=dict(
                gsp_id=batch[GSPSampleKey.gsp_id],
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
            gsp_ids=batch[GSPSampleKey.gsp_id].numpy())

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

    batch_size = config.datamodule.batch_size

    # Prepare DataLoader
    dataloader_kwargs = dict(
        shuffle=False,
        batch_size=batch_size,
        sampler=None,
        batch_sampler=None,
        # num_workers=config.datamodule.num_workers,
        num_workers=2,
        collate_fn=stack_np_samples_into_batch,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        prefetch_factor=config.datamodule.prefetch_factor,
        persistent_workers=False,
    )

    # Get dataset   
    dataset = PVNetUKRegionalDataset(
        config_filename=data_config_path, 
        start_time=start_time, 
        end_time=end_time,
        gsp_ids=ALL_GSP_IDS,
        )

    num_samples = dataset.__len__()

    dataloader = DataLoader(dataset, **dataloader_kwargs)
    # log.info(f"Dataloader created with batch size {config.datamodule.batch_size}, " 
    # f"num_workers {config.datamodule.num_workers}, prefetch factor {config.datamodule.prefetch_factor}")

    # Load GSP capacities as xarray object
    gsp_capacities = get_gsp_capacities(config.datamodule.configuration)

    # Create object - predictions for each input sample
    model_pipe = ModelPipe(model=model, gsp_capacities=gsp_capacities)
    log.info("Model Pipe initialised. Starting inference...")

    # Loop through the samplees
    pbar = tqdm(total=num_samples//batch_size)
    for i, sample in zip(range(num_samples), dataloader):

        da_abs_gsp = model_pipe.predict_sample(sample)
        t0 = da_abs_gsp.init_time_utc.values[0]

        gsp_ids = sample[GSPSampleKey.gsp_id].numpy()
        filename = f"{t0.astype('datetime64[m]')}_batch_{i}_count_{len(gsp_ids)}.nc"

        da_abs_gsp.to_netcdf(f"{output_dir}/{filename}")

        pbar.update()

    pbar.close()
    del dataloader


if __name__ == "__main__":
    main()
