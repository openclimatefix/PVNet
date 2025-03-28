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
from pvnet.models.base_model import BaseModel as PVNetBaseModel

from ocf_data_sampler.numpy_sample.collate import stack_np_samples_into_batch
from ocf_datapipes.batch import batch_to_tensor, copy_batch_to_device, BatchKey, NWPBatchKey
from ocf_data_sampler.config.load import load_yaml_configuration
from ocf_data_sampler.config.save import save_yaml_configuration
from ocf_datapipes.utils.consts import ELEVATION_MEAN, ELEVATION_STD 

from ocf_data_sampler.torch_datasets.datasets.pvnet_uk import PVNetUKConcurrentDataset

import logging

log = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


#_________VARIABLES_________


output_dir = "PLACEHOLDER" # directory where to store predictions

# Time window for which to run the backtest
start_time = "2024-05-31"
end_time = "2024-06-05"

# The frequency at which to create forecast horizons
FREQ_MINS = 30

# When sun at elevation below this, the forecast is set to zero
MIN_DAY_ELEVATION = 0

# All regional GSP IDs
ALL_GSP_IDS = np.arange(1, 318)

#_________MODEL VARIABLES_________

ECMWF_intra = dict(
    hf_model_id = "openclimatefix/pvnet_uk_region",
    hf_revision = "20b882bd4ceaee190a1c994d861f8e5d553ea843", #ECMWF-only pvnet 8h
    hf_summation_model_id = "openclimatefix/pvnet_v2_summation",
    hf_summation_revision = "b40867abbc2e5163c9a665daf511cbf372cc5ac9", #ecmwf-only summation 8h
)

ECMWF_UKV_DA = dict(
    hf_model_id = "openclimatefix/pvnet_uk_region_day_ahead",
    hf_revision = "263741ebb6b71559d113d799c9a579a973cc24ba", #EMCWF+UKV DA
    hf_summation_model_id = "openclimatefix/pvnet_summation_uk_national_day_ahead",
    hf_summation_revision = "7a2f26b94ac261160358b224944ef32998bd60ce"
)

hf_token = None

#_________DATA_PATHS_________

# paths to populate the config with after loading from HF 
ecmwf_path_ens = "PLACEHOLDER-ens.zarr"
ecmwf_path = "PLACEHOLDER.zarr"
gsp_path = "PLACEHOLDER.zarr"
ukv_path = "PLACEHOLDER.zarr"

#_________SELECTION___________
use_ensemble_ecmwf = False
model_dict = ECMWF_UKV_DA


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reformat_config_data_sampler(config: dict) -> dict:
    """Reformat config

    This is to keep the configurations from ocf-data-sampler==0.0.19 working,
    we need to upgrade them a bit to the configuration in ocf-data-sampler>=0.1.5

    Args:
        config: The data config
    """
    # Replace satellite
    if "satellite" in config["input_data"]:

        satellite_config = config["input_data"]["satellite"]

        if satellite_config["satellite_zarr_path"] != "":

            rename_pairs = [
                ("satellite_image_size_pixels_width", "image_size_pixels_width"),
                ("satellite_image_size_pixels_height", "image_size_pixels_height"),
                ("forecast_minutes", "interval_end_minutes"),
                ("satellite_zarr_path", "zarr_path"),
                ("satellite_channels", "channels"),
            ]

            update_config(
                rename_pairs=rename_pairs,
                config=satellite_config,
                remove_keys=["live_delay_minutes"]
            )

    # NWP is nested so must be treated separately
    if "nwp" in config["input_data"]:
        nwp_config = config["input_data"]["nwp"]
        for nwp_source in nwp_config.keys():
            if nwp_config[nwp_source]["nwp_zarr_path"] != "":

                rename_pairs = [
                    ("nwp_image_size_pixels_width", "image_size_pixels_width"),
                    ("nwp_image_size_pixels_height", "image_size_pixels_height"),
                    ("forecast_minutes", "interval_end_minutes"),
                    ("nwp_zarr_path", "zarr_path"),
                    ("nwp_accum_channels", "accum_channels"),
                    ("nwp_channels", "channels"),
                    ("nwp_provider", "provider"),
                ]

                update_config(rename_pairs=rename_pairs, config=nwp_config[nwp_source])

    if "gsp" in config["input_data"]:

        gsp_config = config["input_data"]["gsp"]

        rename_pairs = [
            ("forecast_minutes", "interval_end_minutes"),
            ("gsp_zarr_path", "zarr_path"),
        ]

        update_config(rename_pairs=rename_pairs, config=gsp_config)

    update_config(
        rename_pairs=[],
        config=config["input_data"],
        change_history_minutes=False,
        remove_keys=["default_forecast_minutes", "default_history_minutes"]
    )

    return config


def update_config(rename_pairs: list, config: dict, change_history_minutes: bool = True, remove_keys=None):
    """Update the config in place with rename pairs, and remove keys if they exist

    1. Rename keys in the config
    2. Change history minutes to interval start minutes, with a negative value
    3. Remove keys from the config

    Args:
        rename_pairs: list of pairs to rename
        config: the config dict
        change_history_minutes: option to change history minutes to interval start minutes
        remove_keys: list of key to remove
    """
    for old, new in rename_pairs:
        if old in config:
            config[new] = config[old]
            del config[old]

    if change_history_minutes:
        if "history_minutes" in config:
            config["interval_start_minutes"] = -config["history_minutes"]
            del config["history_minutes"]

    if remove_keys is not None:
        for key in remove_keys:
            if key in config:
                del config[key]


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
    if ukv_path and "ukv" in config["input_data"]["nwp"]:
        config["input_data"]["nwp"]["ukv"]["nwp_zarr_path"] = ukv_path
        log.info(f"Updated ecmwf path to {ecmwf_path}")
    if gsp_path and "gsp" in config["input_data"]:
        config["input_data"]["gsp"]["gsp_zarr_path"] = gsp_path
        log.info(f"Updated gsp path to {gsp_path}")

    return reformat_config_data_sampler(config)


def add_ensemble_member_to_config(data_config_path, ensemble_member: int):
    config = load_yaml_configuration(data_config_path)

    config.input_data.nwp.ecmwf.ensemble_member = ensemble_member
    log.info(f"Updated ECMWF ensemble member to {ensemble_member}")


    config.input_data.nwp.ecmwf.zarr_path = ecmwf_path_ens
    log.info(f"Updated ecmwf path to {ecmwf_path}")

    save_yaml_configuration(configuration=config, filename=data_config_path)


def load_model_from_hf(model_id: str, 
                       revision: str, 
                       token: str,
                       summation: bool = False) -> tuple[Model, str]:
    """Loads model and data config from HuggingFace
    Adapts and saves data config to be used by datasampler. 

    Args:
        model_id: model repo on HF (eg 'openclimatefix/model')
        revision: revision tag for model repo
        token: access token. Needed for access to private repos

    Returns:
        model on device and path to adapted data config
    """

    model = PVNetBaseModel.from_pretrained(
        model_id=model_id,
        revision=revision,
        token=token,
    )

    model.eval().to(device)

    data_config_path = None

    if not summation:
        data_config_file = PVNetBaseModel.get_data_config(
            model_id=model_id,
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
    ds = xr.open_zarr(config.input_data.gsp.zarr_path)

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


def change_keys_to_ocf_datapipes_keys(batch):
    """Change string keys from ocf-data-sampler to BatchKey from ocf-datapipes

    The key change is done in-place

    Until PVNet is merged from dev-data-sampler, we need to do this.
    After this, we might need to change the other way around, for the legacy models.
    """
    keys_to_rename = [
        BatchKey.satellite_actual,
        BatchKey.nwp,
        BatchKey.gsp_solar_elevation,
        BatchKey.gsp_solar_azimuth,
        BatchKey.gsp_id,
        BatchKey.gsp_t0_idx,
        BatchKey.gsp_time_utc,
    ]

    for key in keys_to_rename:
        if key.name in batch:
            batch[key] = batch[key.name]
            del batch[key.name]

    if BatchKey.nwp in batch.keys():
        nwp_batch = batch[BatchKey.nwp]
        for nwp_source in nwp_batch.keys():
            nwp_batch[nwp_source][NWPBatchKey.nwp] = nwp_batch[nwp_source]["nwp"]
            del nwp_batch[nwp_source]["nwp"]


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

    def __init__(self, model: Model, summation_model: Model, gsp_capacities: xr.DataArray):
        """A class to conveniently make and process predictions from batches

        Args:
            model: PVNet GSP level model
            gsp_capacities: xarray dataarray of PVLive capacities
        """
        self.model = model
        self.gsp_capacities = gsp_capacities
        self.summation_model = summation_model

    def predict_sample(self, batch: dict) -> xr.Dataset:
        """Run the batch through the model and compile the predictions into an xarray DataArray

        Args:
            batch: A batch of samples with inputs for each GSP for the same init-time

        Returns:
            xarray.Dataset of all GSP forecasts for the batch
        """

        log.debug("Predicting batch")

        change_keys_to_ocf_datapipes_keys(batch)

        # Unpack some variables from the batch
        id0 = int(batch[BatchKey.gsp_t0_idx])
        t0 = batch[BatchKey.gsp_time_utc].numpy().astype("datetime64[ns]")[0, id0]
        n_valid_times = len(batch[BatchKey.gsp_time_utc][0, id0 + 1 :])
        gsp_capacities = self.gsp_capacities
        model = self.model
        summation_model = self.summation_model

        log.debug(f"Found t0 {t0}. Will forecast for {n_valid_times} points at frequency {FREQ_MINS} minutes")

        # Get valid times for this forecast
        valid_times = pd.to_datetime(
            [t0 + np.timedelta64((i + 1) * FREQ_MINS, "m") for i in range(n_valid_times)]
        )

        log.debug(f"Valid times to forecast for: {valid_times}")

        # Get effective capacities for this forecast
        national_capacity = gsp_capacities.sel(time_utc=t0, gsp_id=0).values
        gsp_capacities = gsp_capacities.sel(
            time_utc=t0, gsp_id=batch[BatchKey.gsp_id]
        ).values

        log.debug(f"GSP capacities identified. Shape: {gsp_capacities.shape}")

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

        log.debug("Prediction ready to save!")

        return da_abs_all


def run_inference(
        config: DictConfig,
        model: Model, 
        summation_model: Model,
        output_dir: str,
        ):
        # Prepare DataLoader
        dataloader_kwargs = dict(
            shuffle=False, # Go through samples t0-first (needs the swap in ocf-data-sampler pvnet_uk_regional L525)
            batch_size=None,
            sampler=None,
            batch_sampler=None,
            num_workers=config.datamodule.num_workers,
            collate_fn=None,
            pin_memory=False,  # Only using CPU to prepare samples so pinning is not beneficial
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
            prefetch_factor=config.datamodule.prefetch_factor,
            persistent_workers=False,  # Not needed since we only enter the dataloader loop once
        )


        # Get the dataset   
        # dataset = PVNetUKRegionalDataset(
        dataset = PVNetUKConcurrentDataset(
            # config_filename=data_config_path,
            config_filename=config.datamodule.configuration, 
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
        model_pipe = ModelPipe(model=model, summation_model=summation_model, gsp_capacities=gsp_capacities)
        log.info("Model Pipe initialised. Starting inference...")

        # Loop through the samplees
        pbar = tqdm(total=num_samples)
        # for i, sample in zip(range(num_samples), dataloader):
        for sample in dataloader:

            # Make predictions for the init-time
            da_abs_gsp = model_pipe.predict_sample(sample)

            # Save the predictions
            t0 = da_abs_gsp.target_datetime_utc.values[int(sample[BatchKey.gsp_t0_idx])]

            filename = f"{t0.astype('datetime64[m]')}.nc"

            da_abs_gsp.to_netcdf(f"{output_dir}/{filename}")

            pbar.update()

        # Close down
        pbar.close()
        del dataloader


@hydra.main(config_path="../configs/", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig) -> None:
    """ 
    Opens model, constructs dataloader, feeds created samples through the model 
    and saves predictions
    """
    
    os.makedirs(f"{output_dir}")

    log.info(
        f"Backtest initiated. Selected time window: {start_time} to {end_time}. "
        f"Total of {len(ALL_GSP_IDS)} GSP IDs requested"
    )
    log.info(f"Device used: {device}")

    # Load data from HuggingFace
    model, data_config_path = load_model_from_hf(
        model_dict["hf_model_id"], 
        model_dict["hf_revision"], 
        hf_token,
        )
    config.datamodule.configuration = data_config_path
    log.info(f"Model loaded from HuggingFace: {model_dict['hf_model_id']}/{model_dict['hf_revision']}")

    if model_dict["hf_summation_model_id"]:
        summation_model, _ = load_model_from_hf(
            model_id=model_dict["hf_summation_model_id"], 
            revision=model_dict["hf_summation_revision"], 
            token=hf_token,
            summation=True,
            )
        log.info(f"Summation model loaded from HuggingFace: {model_dict['hf_summation_model_id']}/{model_dict['hf_summation_revision']}")
    else:
        summation_model = None
        log.info("Summation model not found. Will use sum for national prediction")

    if use_ensemble_ecmwf:
        for i in range(1, 51):
            os.makedirs(f"{output_dir}/ens_{i}")

            log.info(f"Output directory created: {output_dir}/ens_{i}")

            add_ensemble_member_to_config(data_config_path, ensemble_member=i)

            run_inference(config=config, model=model, summation_model=summation_model, output_dir=f"{output_dir}/ens_{i}")
    else:
        os.makedirs(f"{output_dir}/pvnet")
        run_inference(config=config, model=model, summation_model=summation_model, output_dir=f"{output_dir}/pvnet")




if __name__ == "__main__":
    main()
