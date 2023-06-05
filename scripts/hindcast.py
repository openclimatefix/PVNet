"""Script to run hindcasts for a given PVNet model on dates from 2021"""

import logging
import os
import warnings
from datetime import timedelta
from functools import reduce

import numpy as np
import pandas as pd
import torch
import xarray as xr
from ocf_datapipes.load import OpenGSP
from ocf_datapipes.training.pvnet import construct_sliced_data_pipeline
from ocf_datapipes.utils.consts import BatchKey
from ocf_datapipes.utils.utils import stack_np_examples_into_batch
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from torchdata.datapipes.iter import IterableWrapper
from tqdm import tqdm

from pvnet.data.datamodule import batch_to_tensor, copy_batch_to_device
from pvnet.models.base_model import BaseModel
from pvnet.utils import GSPLocationLookup


def get_dataloader_for_loctimes(loc_list, t0_list, num_workers=0, batch_size=None):
    """Get the datalolader for given """
    batch_size = len(loc_list) if batch_size is None else batch_size

    readingservice_config = dict(
        num_workers=num_workers,
        multiprocessing_context="spawn",
        worker_prefetch_cnt=0 if num_workers == 0 else 2,
    )

    # This iterates though all times for loc_list[0] before moving on to loc_list[1]
    # This stops us wasting time if some timestamp in the day has missing values
    location_pipe = IterableWrapper(loc_list).repeat(len(t0_list))
    t0_datapipe = IterableWrapper(t0_list).cycle(len(loc_list))

    location_pipe = location_pipe.sharding_filter()
    t0_datapipe = t0_datapipe.sharding_filter()

    batch_pipe = construct_sliced_data_pipeline(
        config_filename="../configs/datamodule/configuration/gcp_configuration.yaml",
        location_pipe=location_pipe,
        t0_datapipe=t0_datapipe,
        block_sat=False,
        block_nwp=False,
    )

    batch_pipe = (
        batch_pipe.batch(batch_size=batch_size)
        .map(stack_np_examples_into_batch)
        .map(batch_to_tensor)
    )

    rs = MultiProcessingReadingService(**readingservice_config)
    dataloader = DataLoader2(batch_pipe, reading_service=rs)
    return dataloader


def save_date_preds(x, date, path_root):
    """Save the predictions for date to zarr"""
    a = np.zeros((len(x.keys()), 317, 16))

    for i, k in enumerate(list(x.keys())):
        df = pd.DataFrame(x[k])[np.arange(1, 318)]
        a[i] = df.values.T

    ds = xr.DataArray(
        a,
        dims=["t0_time", "gsp_id", "step"],
        coords=dict(t0_time=list(x.keys()), gsp_id=np.arange(1, 318), step=np.arange(16)),
    ).to_dataset(name="preds")
    ds.to_zarr(f"{path_root}/{date}.zarr")


if __name__ == "__main__":
    # ---------------------------------------------------------------------------
    # User params

    path_root = "/mnt/disks/batches2/may-july_hindcast"
    model_name = "openclimatefix/pvnet_v2"
    model_version = "7cc7e9f8e5fc472a753418c45b2af9f123547b6c"

    # We assume we only ever look at 2021
    # If set to None this sctipt will find all days where we have input data
    # date_range = ("2021-01-01", "2021-01-02") - will give 2 days
    date_range = ("2021-04-19", "2021-06-19")

    gsp_ids = np.arange(1, 318)

    times = [timedelta(minutes=i * 30) for i in range(48)]
    batch_size = 10
    num_workers = 20

    # ---------------------------------------------------------------------------
    # Initial set-up

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.info("Setting up")

    # Set up directory first in case of path already existing error
    os.makedirs(path_root, exist_ok=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_gsp = next(iter(OpenGSP("gs://solar-pv-nowcasting-data/PV/GSP/v5/pv_gsp.zarr")))

    # Set up ID location query object
    gsp_id_to_loc = GSPLocationLookup(ds_gsp.x_osgb, ds_gsp.y_osgb)

    location_list = [gsp_id_to_loc(gsp_id) for gsp_id in gsp_ids]

    # ---------------------------------------------------------------------------
    # Construct array of dates

    logger.info("Constructing date arrays")

    if date_range is None:
        ds_nwp = xr.open_zarr("/mnt/disks/nwp/UKV_intermediate_version_7.zarr")

        ds_sat = xr.open_zarr("/mnt/disks/data_ssd/2021_nonhrv.zarr")

        potential_dates = reduce(
            np.intersect1d,
            [
                np.unique(ds_nwp.init_time.dt.date),
                np.unique(ds_sat.time.dt.date),
                np.unique(ds_gsp.time_utc.dt.date),
            ],
        )
        del ds_sat, ds_nwp

    else:
        potential_dates = pd.date_range(*date_range, freq=timedelta(days=1)).date

    # ---------------------------------------------------------------------------
    # Load the model

    logger.info("Loading model")

    model = BaseModel.from_pretrained(model_name, revision=model_version)
    model = model.to(device)
    model = model.eval()

    # ---------------------------------------------------------------------------
    # Run

    logger.info("Beginning hindcasts")

    pbar = tqdm(total=len(potential_dates) * len(times) * len(location_list))

    # Expected n on pbar after next date iteration
    # Store this in case some dates fail. Allows pbar to be kept up to dat regardless of failure on
    # some dates.
    pbar_n = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for date in potential_dates:
            pbar_n += len(location_list) * len(times)

            t0_list = [np.datetime64(pd.Timestamp(date) + dt, "s") for dt in times]

            date_preds = {t0: dict() for t0 in t0_list}

            dataloader = get_dataloader_for_loctimes(
                loc_list=location_list,
                t0_list=t0_list,
                num_workers=num_workers,
                batch_size=batch_size,
            )
            # We lump all times in a day together. We either complete the entire day of forecasts or
            # fail on the entire day
            try:
                for i, batch in enumerate(dataloader):
                    with torch.no_grad():
                        preds = model(copy_batch_to_device(batch, device)).detach().cpu().numpy()

                        batch_times = (
                            (batch[BatchKey.gsp_time_utc][:, batch[BatchKey.gsp_t0_idx]])
                            .numpy()
                            .astype("datetime64[s]")
                        )

                        for id, pred, time in zip(batch[BatchKey.gsp_id], preds, batch_times):
                            if id in date_preds[time]:
                                logger.warning(
                                    f"ID {id} already exists in entry for datetime {time}"
                                )
                            date_preds[time][id.item()] = pred
                            pbar.update()
                    should_save = True
            except Exception:
                logger.exception(f"Date: {date} failed")
                # Round up the progress bar
                pbar.update(pbar_n - pbar.n)
                should_save = False

            # This gives a hacky way to stop this program. Deleting the output dir will cause it to
            # error out
            if should_save:
                save_date_preds(date_preds, date, path_root=path_root)

    pbar.close()
    # ---------------------------------------------------------------------------
    # Consolidate up all the zarr stores
    ds = xr.open_mfdataset(f"{path_root}/*.zarr", engine="zarr").compute()
    ds.to_zarr(f"{path_root}.zarr")

    os.system(f"rm -r {path_root}/*.zarr")
    os.system(f"rmdir {path_root}")

    logger.info("Hindcast complete")
