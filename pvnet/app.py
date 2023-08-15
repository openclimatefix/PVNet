"""App to run inference

This app expects these evironmental variables to be available:
    - DB_URL
    - NWP_ZARR_PATH
    - SATELLITE_ZARR_PATH
"""

import logging
import os
import warnings
from datetime import datetime, timedelta, timezone

import fsspec
import numpy as np
import pandas as pd
import torch
import typer
import xarray as xr
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.models import (
    ForecastSQL,
    ForecastValue,
)
from nowcasting_datamodel.read.read import (
    get_latest_input_data_last_updated,
    get_location,
    get_model,
)
from nowcasting_datamodel.save.save import save as save_sql_forecasts
from ocf_datapipes.load import OpenGSPFromDatabase
from ocf_datapipes.training.pvnet import construct_sliced_data_pipeline
from ocf_datapipes.transform.numpy.batch.sun_position import ELEVATION_MEAN, ELEVATION_STD
from ocf_datapipes.utils.consts import BatchKey
from ocf_datapipes.utils.utils import stack_np_examples_into_batch
from pvnet_summation.models.base_model import BaseModel as SummationBaseModel
from sqlalchemy.orm import Session
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from torchdata.datapipes.iter import IterableWrapper

import pvnet
from pvnet.data.datamodule import batch_to_tensor, copy_batch_to_device
from pvnet.models.base_model import BaseModel as PVNetBaseModel
from pvnet.utils import GSPLocationLookup

# ---------------------------------------------------------------------------
# GLOBAL SETTINGS

# TODO: Host data config alongside model?
this_dir = os.path.dirname(os.path.abspath(__file__))

data_config_filename = f"{this_dir}/../configs/datamodule/configuration/app_configuration.yaml"

# Model will use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# If the solar elevation is less than this the predictions are set to zero
MIN_DAY_ELEVATION = 0

# Forecast made for these GSP IDs and summed to national with ID=>0
all_gsp_ids = list(range(1, 318))

# Batch size used to make forecasts for all GSPs
batch_size = 10

# Huggingfacehub model repo and commit for PVNet (GSP-level model)
model_name = os.getenv("APP_MODEL", default="openclimatefix/pvnet_v2")
model_version = os.getenv("APP_MODEL_VERSION", default="96ac8c67fa8663844ddcfa82aece51ef94f34453")

# Huggingfacehub model repo and commit for PVNet summation (GSP sum to national model)
# If summation_model_name is set to None, a simple sum is computed instead
summation_model_name = os.getenv("APP_SUMMATION_MODEL", default="openclimatefix/pvnet_v2_summation")
summation_model_version = os.getenv(
    "APP_SUMMATION_MODEL", default="4a145d74c725ffc72f482025d3418659a6869c94"
)


model_name_ocf_db = "pvnet_v2"
use_adjuster = os.getenv("USE_ADJUSTER", "True").lower() == "true"

# ---------------------------------------------------------------------------
# LOGGER
formatter = logging.Formatter(
    fmt="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s"
)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, os.getenv("LOGLEVEL", "INFO")))
logger.addHandler(stream_handler)

# Get rid of these verbose logs
sql_logger = logging.getLogger("sqlalchemy.engine.Engine")
sql_logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS


def convert_dataarray_to_forecasts(
    forecast_values_dataarray: xr.DataArray, session: Session, model_name: str, version: str
) -> list[ForecastSQL]:
    """
    Make a ForecastSQL object from a DataArray.

    Args:
        forecast_values_dataarray: Dataarray of forecasted values. Must have `target_datetime_utc`
            `gsp_id`, and `output_label` coords. The `output_label` coords must have `"forecast_mw"`
            as an element.
        session: database session
        model_name: the name of the model
        version: the version of the model
    Return:
        List of ForecastSQL objects
    """
    logger.debug("Converting DataArray to list of ForecastSQL")

    assert "target_datetime_utc" in forecast_values_dataarray.coords
    assert "gsp_id" in forecast_values_dataarray.coords
    assert "forecast_mw" in forecast_values_dataarray.output_label

    # get last input data
    input_data_last_updated = get_latest_input_data_last_updated(session=session)

    # get model name
    model = get_model(name=model_name, version=version, session=session)

    forecasts = []

    for gsp_id in forecast_values_dataarray.gsp_id.values:
        gsp_id = int(gsp_id)
        # make forecast values
        forecast_values = []

        # get location
        location = get_location(session=session, gsp_id=gsp_id)

        gsp_forecast_values_da = forecast_values_dataarray.sel(gsp_id=gsp_id)

        for target_time in pd.to_datetime(gsp_forecast_values_da.target_datetime_utc.values):
            # add timezone
            target_time_utc = target_time.replace(tzinfo=timezone.utc)
            this_da = gsp_forecast_values_da.sel(target_datetime_utc=target_time)

            forecast_value_sql = ForecastValue(
                target_time=target_time_utc,
                expected_power_generation_megawatts=(
                    this_da.sel(output_label="forecast_mw").item()
                ),
            ).to_orm()

            forecast_value_sql.adjust_mw = 0.0

            forecast_value_sql.properties = {}

            if "forecast_mw_plevel_10" in gsp_forecast_values_da.output_label:
                val = this_da.sel(output_label="forecast_mw_plevel_10").item()
                # `val` can be NaN if PVNet has probabilistic outputs and PVNet_summation doesn't,
                # or if PVNet_summation has probabilistic outputs and PVNet doesn't.
                # Do not log the value if NaN
                if not np.isnan(val):
                    forecast_value_sql.properties["10"] = val

            if "forecast_mw_plevel_90" in gsp_forecast_values_da.output_label:
                val = this_da.sel(output_label="forecast_mw_plevel_90").item()

                if not np.isnan(val):
                    forecast_value_sql.properties["90"] = val

            forecast_values.append(forecast_value_sql)

        # make forecast object
        forecast = ForecastSQL(
            model=model,
            forecast_creation_time=datetime.now(tz=timezone.utc),
            location=location,
            input_data_last_updated=input_data_last_updated,
            forecast_values=forecast_values,
            historic=False,
        )

        forecasts.append(forecast)

    return forecasts


def app(
    t0=None,
    apply_adjuster: bool = use_adjuster,
    gsp_ids: list[int] = all_gsp_ids,
    write_predictions: bool = True,
    num_workers: int = -1,
):
    """Inference function for production

    This app expects these evironmental variables to be available:
        - DB_URL
        - NWP_ZARR_PATH
        - SATELLITE_ZARR_PATH
    Args:
        t0 (datetime): Datetime at which forecast is made
        apply_adjuster (bool): Whether to apply the adjuster when saving forecast
        gsp_ids (array_like): List of gsp_ids to make predictions for. This list of GSPs are summed
            to national.
        write_predictions (bool): Whether to write prediction to the database. Else returns as
            DataArray for local testing.
        num_workers (int): Number of workers to use to load batches of data. When set to default
            value of -1, it will use one less than the number of CPU cores workers.
    """

    if num_workers == -1:
        num_workers = os.cpu_count() - 1

    logger.info(f"Using `pvnet` library version: {pvnet.__version__}")
    logger.info(f"Using {num_workers} workers")

    # ---------------------------------------------------------------------------
    # 0. If inference datetime is None, round down to last 30 minutes
    if t0 is None:
        t0 = pd.Timestamp.now(tz="UTC").replace(tzinfo=None).floor(timedelta(minutes=30))
    else:
        t0 = pd.to_datetime(t0).floor(timedelta(minutes=30))

    if len(gsp_ids) == 0:
        gsp_ids = all_gsp_ids

    logger.info(f"Making forecast for init time: {t0}")
    logger.info(f"Making forecast for GSP IDs: {gsp_ids}")

    # ---------------------------------------------------------------------------
    # 1. Prepare data sources

    # Make pands Series of most recent GSP effective capacities

    logger.info("Loading GSP metadata")

    ds_gsp = next(iter(OpenGSPFromDatabase()))

    # DataArray of most recent GSP capacities
    gsp_capacities = (
        ds_gsp.sel(
            time_utc=t0,
            method="ffill",
        )
        .sel(gsp_id=gsp_ids)
        .reset_coords()
        .effective_capacity_mwp
    )

    # National capacity is needed if using summation model
    ds_gsp_national = next(iter(OpenGSPFromDatabase(national_only=True)))
    national_capacity = ds_gsp_national.sel(
        time_utc=t0, method="ffill"
    ).effective_capacity_mwp.item()

    # Set up ID location query object
    gsp_id_to_loc = GSPLocationLookup(ds_gsp.x_osgb, ds_gsp.y_osgb)

    # Download satellite data - can't load zipped zarr straight from s3 bucket
    logger.info("Downloading zipped satellite data")
    fs = fsspec.open(os.environ["SATELLITE_ZARR_PATH"]).fs
    fs.get(os.environ["SATELLITE_ZARR_PATH"], "latest.zarr.zip")

    # Also download 15-minute satellite if it exists
    sat_latest_15 = os.environ["SATELLITE_ZARR_PATH"].replace(".zarr.zip", "_15.zarr.zip")
    if fs.exists(sat_latest_15):
        logger.info("Downloading 15-minute satellite data")
        fs.get(sat_latest_15, "latest_15.zarr.zip")

    # ---------------------------------------------------------------------------
    # 2. Set up data loader
    logger.info("Creating DataLoader")

    # Location and time datapipes
    location_pipe = IterableWrapper([gsp_id_to_loc(gsp_id) for gsp_id in gsp_ids])
    t0_datapipe = IterableWrapper([t0]).repeat(len(location_pipe))

    location_pipe = location_pipe.sharding_filter()
    t0_datapipe = t0_datapipe.sharding_filter()

    # Batch datapipe
    batch_datapipe = (
        construct_sliced_data_pipeline(
            config_filename=data_config_filename,
            location_pipe=location_pipe,
            t0_datapipe=t0_datapipe,
            production=True,
            check_satellite_no_zeros=True,
        )
        .batch(batch_size)
        .map(stack_np_examples_into_batch)
    )

    # Set up dataloader for parallel loading
    rs = MultiProcessingReadingService(
        num_workers=num_workers,
        multiprocessing_context="spawn",
        worker_prefetch_cnt=0 if num_workers == 0 else 2,
    )
    dataloader = DataLoader2(batch_datapipe, reading_service=rs)

    # ---------------------------------------------------------------------------
    # 3. set up model
    logger.info(f"Loading model: {model_name} - {model_version}")

    model = PVNetBaseModel.from_pretrained(
        model_name,
        revision=model_version,
    ).to(device)

    if summation_model_name is not None:
        summation_model = SummationBaseModel.from_pretrained(
            summation_model_name,
            revision=summation_model_version,
        ).to(device)

        if (
            summation_model.pvnet_model_name != model_name
            or summation_model.pvnet_model_version != model_version
        ):
            warnings.warn(
                f"The PVNet version running in this app is "
                f"{model_name}/{model_version}."
                f"The summation model running in this app was trained on outputs from PVNet "
                f"version {summation_model.model_name}/{summation_model.model_version}. "
                f"Combining these models may lead to an error if the shape of PVNet output doesn't "
                f"match the expected shape of the summation model. Combining may lead to "
                f"unreliable results even if the shapes match."
            )

    # 4. Make prediction
    logger.info("Processing batches")
    normed_preds = []
    gsp_ids_each_batch = []
    sun_down_masks = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            logger.info(f"Predicting for batch: {i}")

            # Store GSP IDs for this batch for reordering later
            these_gsp_ids = batch[BatchKey.gsp_id]
            gsp_ids_each_batch += [these_gsp_ids]

            # Run batch through model
            device_batch = copy_batch_to_device(batch_to_tensor(batch), device)
            preds = model(device_batch).detach().cpu().numpy()

            # Calculate unnormalised elevation and sun-dowm mask
            logger.info("Zeroing predictions after sundown")
            elevation = batch[BatchKey.gsp_solar_elevation] * ELEVATION_STD + ELEVATION_MEAN
            # We only need elevation mask for forecasted values, not history
            elevation = elevation[:, -preds.shape[1] :]
            sun_down_mask = elevation < MIN_DAY_ELEVATION

            # Store predictions
            normed_preds += [preds]
            sun_down_masks += [sun_down_mask]

            # log max prediction
            logger.info(f"GSP IDs: {these_gsp_ids}")
            logger.info(f"Max prediction: {np.max(preds, axis=1)}")
            logger.info(f"Completed batch: {i}")

    normed_preds = np.concatenate(normed_preds)
    sun_down_masks = np.concatenate(sun_down_masks)

    gsp_ids_all_batches = np.concatenate(gsp_ids_each_batch).squeeze()

    # Reorder GSP order which ends up shuffled if multiprocessing is used
    inds = gsp_ids_all_batches.argsort()

    normed_preds = normed_preds[inds]
    sun_down_masks = sun_down_masks[inds]
    gsp_ids_all_batches = gsp_ids_all_batches[inds]

    logger.info(f"{gsp_ids_all_batches.shape}")

    # ---------------------------------------------------------------------------
    # 5. Merge batch results to xarray DataArray
    logger.info("Processing raw predictions to DataArray")

    n_times = normed_preds.shape[1]

    if model.use_quantile_regression:
        output_labels = model.output_quantiles
        output_labels = [f"forecast_mw_plevel_{int(q*100):02}" for q in model.output_quantiles]
        output_labels[output_labels.index("forecast_mw_plevel_50")] = "forecast_mw"
    else:
        output_labels = ["forecast_mw"]
        normed_preds = normed_preds[..., np.newaxis]

    da_normed = xr.DataArray(
        data=normed_preds,
        dims=["gsp_id", "target_datetime_utc", "output_label"],
        coords=dict(
            gsp_id=gsp_ids_all_batches,
            target_datetime_utc=pd.to_datetime(
                [t0 + timedelta(minutes=30 * (i + 1)) for i in range(n_times)],
            ),
            output_label=output_labels,
        ),
    )

    da_sundown_mask = xr.DataArray(
        data=sun_down_masks,
        dims=["gsp_id", "target_datetime_utc"],
        coords=dict(
            gsp_id=gsp_ids_all_batches,
            target_datetime_utc=pd.to_datetime(
                [t0 + timedelta(minutes=30 * (i + 1)) for i in range(n_times)],
            ),
        ),
    )

    # Multiply normalised forecasts by capacities and clip negatives
    logger.info(f"Converting to absolute MW using {gsp_capacities}")
    da_abs = da_normed.clip(0, None) * gsp_capacities
    max_preds = da_abs.sel(output_label="forecast_mw").max(dim="target_datetime_utc")
    logger.info(f"Maximum predictions: {max_preds}")

    # Apply sundown mask
    da_abs = da_abs.where(~da_sundown_mask).fillna(0.0)

    # ---------------------------------------------------------------------------
    # 6. Make national total
    logger.info("Summing to national forecast")

    if summation_model_name is not None:
        logger.info("Using summation model to produce national forecast")

        # Make national predictions using summation model
        inputs = {
            "pvnet_outputs": torch.Tensor(normed_preds[np.newaxis]).to(device),
            "effective_capacity": (
                torch.Tensor(gsp_capacities.values / national_capacity)
                .to(device)
                .unsqueeze(0)
                .unsqueeze(-1)
            ),
        }
        normed_national = summation_model(inputs).detach().squeeze().cpu().numpy()

        # Convert national predictions to DataArray
        if summation_model.use_quantile_regression:
            sum_output_labels = summation_model.output_quantiles
            sum_output_labels = [
                f"forecast_mw_plevel_{int(q*100):02}" for q in summation_model.output_quantiles
            ]
            sum_output_labels[sum_output_labels.index("forecast_mw_plevel_50")] = "forecast_mw"
        else:
            sum_output_labels = ["forecast_mw"]

        da_normed_national = xr.DataArray(
            data=normed_national[np.newaxis],
            dims=["gsp_id", "target_datetime_utc", "output_label"],
            coords=dict(
                gsp_id=[0],
                target_datetime_utc=da_abs.target_datetime_utc,
                output_label=sum_output_labels,
            ),
        )

        # Multiply normalised forecasts by capacities and clip negatives
        da_abs_national = da_normed_national.clip(0, None) * national_capacity

        # Apply sundown mask - All GSPs must be masked to mask national
        da_abs_national = da_abs_national.where(~da_sundown_mask.all(dim="gsp_id")).fillna(0.0)

        da_abs_all = xr.concat([da_abs_national, da_abs], dim="gsp_id")

    else:
        logger.info("Summing across GSPs to produce national forecast")
        da_abs_national = (
            da_abs.sum(dim="gsp_id").expand_dims(dim="gsp_id", axis=0).assign_coords(gsp_id=[0])
        )
        da_abs_all = xr.concat([da_abs_national, da_abs], dim="gsp_id")
        logger.info(
            f"National forecast is {da_abs.sel(gsp_id=0, output_label='forecast_mw').values}"
        )

    # ---------------------------------------------------------------------------
    # Escape clause for making predictions locally
    if not write_predictions:
        return da_abs_all

    # ---------------------------------------------------------------------------
    # 7. Write predictions to database
    logger.info("Writing to database")

    connection = DatabaseConnection(url=os.environ["DB_URL"])
    with connection.get_session() as session:
        sql_forecasts = convert_dataarray_to_forecasts(
            da_abs_all, session, model_name=model_name_ocf_db, version=pvnet.__version__
        )

        save_sql_forecasts(
            forecasts=sql_forecasts,
            session=session,
            update_national=True,
            update_gsp=True,
            apply_adjuster=apply_adjuster,
        )

    logger.info("Finished forecast")


if __name__ == "__main__":
    typer.run(app)
