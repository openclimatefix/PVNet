"""Script to run the production app locally"""

import logging
import os

import pandas as pd
import numpy as np
import xarray as xr

from datetime import timedelta
import time

from ocf_datapipes.load import OpenGSPFromDatabase
from pvnet.app import app


formatter = logging.Formatter(fmt="%(levelname)s:%(name)s:%(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(stream_handler)


def sleep_until(wake_time):
    now = pd.Timestamp.now()
    sleep_duration = (wake_time-now).total_seconds()
    if sleep_duration<0:
        logger.warning("Sleep for negative duration requested")
    else:
        logger.info(f"Sleeping for {sleep_duration} seconds")
        time.sleep(sleep_duration)


if __name__=="__main__":
    
    # ----------------------------------------------------
    # USER SETTINGS
    
    # When to start and stop predictions
    start_time = pd.Timestamp("2023-05-31 00:00")
    end_time = pd.Timestamp("2023-06-05 21:00")
    
    output_dir = "/mnt/disks/batches/local_production_forecasts"
    save_inputs = True
    
    # ----------------------------------------------------
    # RUN
    
    # Make output dirs
    os.makedirs(f"{output_dir}/predictions", exist_ok=True)
    os.makedirs(f"{output_dir}/logs", exist_ok=True)
    if save_inputs:
        os.makedirs(f"{output_dir}/inputs", exist_ok=True)
    
    
    # Wait until start time
    if pd.Timestamp.now() < start_time:
        sleep_until(start_time)
    
    while pd.Timestamp.now() < end_time:
        
        # Next prediction time
        t0 =  pd.Timestamp.now().ceil(timedelta(minutes=30))
        
        # Sleep until next prediction time
        sleep_until(t0)

        try:
            # Make predictions
            df = app(write_predictions=False)

            # Save
            df.to_csv(f"{output_dir}/predictions/{t0}.csv")
        except:
            logger.exception(f"Predictions for {t0=} failed")

        try:
            # Log delays of data sources
            log = dict(
                now=t0,
                gsp_times=next(iter(OpenGSPFromDatabase())).time_utc.values,
                sat_times=xr.open_zarr("latest.zarr.zip").time.values,
                nwp_times=xr.open_zarr(os.environ['NWP_ZARR_PATH']).init_time.values,
            )
            np.save(f"{output_dir}/logs/{t0}.npy",  log)
        except:
            logger.exception(f"Logs for {t0=} failed")
            
        if save_inputs:
            try:
                # Set up directory to save inputs
                input_dir = f"{output_dir}/inputs/{t0}"
                os.makedirs(input_dir, exist_ok=True)

                # Save inputs
                os.system(f"cp latest.zarr.zip '{input_dir}/sat.zarr.zip'")
                
                ds = xr.open_zarr(os.environ['NWP_ZARR_PATH'])
                for v in ds.variables:
                    ds[v].encoding.clear()
                ds.to_zarr(f"{input_dir}/nwp.zarr")
                
                next(iter(OpenGSPFromDatabase())).to_dataset().to_zarr(f"{input_dir}/gsp.zarr")

            except:
                logger.exception(f"Saving inputs for {t0=} failed")
