"""App to run inference

This app expects these evironmental variables to be available:
    - DB_URL
    - NWP_ZARR_PATH
    - SATELLITE_ZARR_PATH
"""

import logging
import os
from datetime import datetime, timedelta, timezone

import fsspec
import numpy as np
import pandas as pd
import torch
import typer
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
from pvlive_api import PVLive
from sqlalchemy.orm import Session
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from torchdata.datapipes.iter import IterableWrapper

import pvnet
from pvnet.data.datamodule import batch_to_tensor, copy_batch_to_device
from pvnet.models.base_model import BaseModel
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

# Huggingfacehub model repo and commit
model_name = "openclimatefix/pvnet_v2"
model_version = "3d01665c89501da25e3acaaee6b669a15748708c"

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


def convert_df_to_forecasts(
    forecast_values_df: pd.DataFrame, session: Session, model_name: str, version: str
) -> list[ForecastSQL]:
    """
    Make a ForecastSQL object from a dataframe.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        forecast_values_df (Dataframe): Containing `target_datetime_utc` and `forecast_mw` columns
        session: database session
        model_name: the name of the model
        version: the version of the model
    Return:
        List of ForecastSQL objects
    """

    logger.debug("Converting dataframe to list of ForecastSQL")

    assert "target_datetime_utc" in forecast_values_df.columns
    assert "forecast_mw" in forecast_values_df.columns
    assert "gsp_id" in forecast_values_df.columns

    # get last input data
    input_data_last_updated = get_latest_input_data_last_updated(session=session)

    # get model name
    model = get_model(name=model_name, version=version, session=session)

    forecasts = []

    for gsp_id in forecast_values_df.gsp_id.unique():
        # make forecast values
        forecast_values = []

        # get location
        location = get_location(session=session, gsp_id=gsp_id)

        gsp_forecast_values_df = forecast_values_df.query(f"gsp_id=={gsp_id}")

        for i, forecast_value in gsp_forecast_values_df.iterrows():
            # add timezone
            target_time = forecast_value.target_datetime_utc.replace(tzinfo=timezone.utc)
            forecast_value_sql = ForecastValue(
                target_time=target_time,
                expected_power_generation_megawatts=forecast_value.forecast_mw,
            ).to_orm()
            forecast_value_sql.adjust_mw = 0.0
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
            DataFrame for local testing.
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

    # Pulling these valiues from the PVLive API is a temporary fix until capacity_mwp is available
    # in the database. This should be removed/updated once available.
    logger.info("Requesting GSP effective capacities from PVLive")

    gsp_capacities = pd.Series(
        np.zeros(len(gsp_ids), dtype=np.float32),
        name="capacity_megawatt_power",
        index=pd.Series(gsp_ids, name="gsp_id"),
    )

    # Take capacities at most recent midnight
    capacity_time = datetime.now(tz=timezone.utc).replace(hour=0, minute=0, second=0)

    pvl = PVLive()

    for gsp_id in gsp_ids:
        cap_df = pvl.between(
            start=capacity_time,
            end=capacity_time,
            entity_type="gsp",
            entity_id=gsp_id,
            extra_fields="capacity_mwp",
            dataframe=True,
        )

        gsp_capacities.loc[gsp_id] = cap_df.capacity_mwp.item()

    logger.info("Loading GSP metadata")

    ds_gsp = next(iter(OpenGSPFromDatabase()))

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

    model = BaseModel.from_pretrained(
        os.getenv("APP_MODEL", default=model_name),
        revision=os.getenv("APP_MODEL_VERSION", default=model_version),
    ).to(device)

    # 4. Make prediction
    logger.info("Processing batches")
    normed_preds = []
    gsp_ids_each_batch = []

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
            elevation = elevation[:, -preds.shape[-1] :]
            sun_down_mask = elevation < MIN_DAY_ELEVATION

            # Zero out after sundown
            preds[sun_down_mask] = 0

            # Store predictions
            normed_preds += [preds]

            # log max prediction
            logger.info(f"GSP IDs: {these_gsp_ids}")
            logger.info(f"Max prediction: {np.max(preds, axis=1)}")
            logger.info(f"Completed batch: {i}")

    normed_preds = np.concatenate(normed_preds)
    gsp_ids_all_batches = np.concatenate(gsp_ids_each_batch).squeeze()
    logger.info(f"{gsp_ids_all_batches.shape}")

    # ---------------------------------------------------------------------------
    # 5. Merge batch results to pandas df
    logger.info("Processing raw predictions to DataFrame")

    n_times = normed_preds.shape[1]

    df_normed = pd.DataFrame(
        normed_preds.T,
        columns=gsp_ids_all_batches,
        index=pd.Index(
            [t0 + timedelta(minutes=30 * (i + 1)) for i in range(n_times)],
            name="target_datetime_utc",
        ),
    )

    # Reorder columns which end up shuffled if multiprocessing is used
    df_normed = df_normed.loc[:, gsp_ids]

    # Multiply normalised forecasts by capacities and clip negatives
    logger.info(f"Converting to absolute MW using {gsp_capacities}")
    df_abs = df_normed.clip(0, None) * gsp_capacities.T
    logger.info(f"Maximum predictions: {df_abs.max()}")

    # ---------------------------------------------------------------------------
    # 6. Make national total
    logger.info("Summing to national forecast")
    df_abs.insert(0, 0, df_abs.sum(axis=1))
    logger.info(f"National forecast is {df_abs[0]}")

    # ---------------------------------------------------------------------------
    # Escape clause for making predictions locally
    if not write_predictions:
        return df_abs

    # ---------------------------------------------------------------------------
    # 7. Write predictions to database
    logger.info("Writing to database")

    # Flatten DataFrame
    df = df_abs.reset_index().melt(
        "target_datetime_utc",
        var_name="gsp_id",
        value_name="forecast_mw",
    )

    connection = DatabaseConnection(url=os.environ["DB_URL"])
    with connection.get_session() as session:
        sql_forecasts = convert_df_to_forecasts(
            df, session, model_name=model_name_ocf_db, version=pvnet.__version__
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
