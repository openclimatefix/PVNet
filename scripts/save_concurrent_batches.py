"""
Constructs batches where each batch includes all GSPs and each sample within a batch has the same
timestamp.

Currently a slightly hacky implementation due to the way the configs are done. This script will use
the same config file currently set to train the model.

use:
```
python save_concurrent_batches.py \
    +batch_output_dir="/mnt/disks/batches/concurrent_batches_v0" \
    +num_train_batches=1_000 \
    +num_val_batches=200
```

"""

import logging
import os
import shutil
import sys
import warnings

import hydra
import numpy as np
import torch
from ocf_datapipes.training.common import (
    open_and_return_datapipes,
)
from ocf_datapipes.training.pvnet import construct_loctime_pipelines, construct_sliced_data_pipeline
from ocf_datapipes.utils.consts import BatchKey
from ocf_datapipes.utils.utils import stack_np_examples_into_batch
from omegaconf import DictConfig, OmegaConf
from sqlalchemy import exc as sa_exc
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from torchdata.datapipes.iter import IterableWrapper
from tqdm import tqdm

from pvnet.data.datamodule import batch_to_tensor
from pvnet.utils import GSPLocationLookup

warnings.filterwarnings("ignore", category=sa_exc.SAWarning)

logger = logging.getLogger(__name__)

logging.basicConfig(stream=sys.stdout, level=logging.ERROR)


class _save_batch_func_factory:
    def __init__(self, batch_dir):
        self.batch_dir = batch_dir

    def __call__(self, input):
        i, batch = input
        torch.save(batch, f"{self.batch_dir}/{i:06}.pt")


def _get_datapipe(config_path, start_time, end_time, n_batches):
    # Open datasets from the config and filter to useable location-time pairs
    _, t0_datapipe = construct_loctime_pipelines(
        config_path,
        start_time,
        end_time,
    )

    # Set up ID location query object
    ds_gsp = next(
        iter(
            open_and_return_datapipes(
                config_path,
                use_gsp=True,
                use_nwp=False,
                use_pv=False,
                use_sat=False,
                use_hrv=False,
                use_topo=False,
            )["gsp"]
        )
    )
    gsp_id_to_loc = GSPLocationLookup(ds_gsp.x_osgb, ds_gsp.y_osgb)

    # Cycle the GSP locations
    location_pipe = IterableWrapper([[gsp_id_to_loc(gsp_id) for gsp_id in range(1, 318)]]).repeat(
        n_batches
    )

    # Shard and unbatch so each worker goes through GSP 1-317 for each batch
    location_pipe = location_pipe.sharding_filter()
    location_pipe = location_pipe.unbatch(unbatch_level=1)

    # Shard and repeat so each worker repeats the same time for the entire batch
    t0_datapipe = t0_datapipe.sharding_filter()
    t0_datapipe = t0_datapipe.repeat(317)

    data_pipeline = construct_sliced_data_pipeline(
        config_path,
        location_pipe,
        t0_datapipe,
    )

    data_pipeline = data_pipeline.batch(317).map(stack_np_examples_into_batch).map(batch_to_tensor)

    return data_pipeline


def _save_batches_with_dataloader(batch_pipe, batch_dir, num_batches, rs_config):
    save_func = _save_batch_func_factory(batch_dir)
    filenumber_pipe = IterableWrapper(range(num_batches)).sharding_filter()
    save_pipe = filenumber_pipe.zip(batch_pipe).map(save_func)

    rs = MultiProcessingReadingService(**rs_config)
    dataloader = DataLoader2(save_pipe, reading_service=rs)

    pbar = tqdm(total=num_batches)
    for i, batch in zip(range(num_batches), dataloader):
        pbar.update()
    pbar.close()
    del dataloader


def check_batch(batch):
    """Check if batch is valid concurrent batch for all GSPs"""
    # Check all GSP IDs are included and in correct order
    assert (batch[BatchKey.gsp_id].flatten().numpy() == np.arange(1, 318)).all()
    # Check all times are the same
    assert len(np.unique(batch[BatchKey.gsp_time_utc][:, 0].numpy())) == 1
    return batch


@hydra.main(config_path="../configs/", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig):
    """Constructs and saves validation and training batches."""
    config_dm = config.datamodule

    # Set up directory
    os.makedirs(config.batch_output_dir, exist_ok=False)

    with open(f"{config.batch_output_dir}/datamodule.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(config.datamodule))

    shutil.copyfile(config_dm.configuration, f"{config.batch_output_dir}/data_configuration.yaml")

    os.mkdir(f"{config.batch_output_dir}/train")
    os.mkdir(f"{config.batch_output_dir}/val")

    readingservice_config = dict(
        num_workers=config_dm.num_workers,
        multiprocessing_context="spawn",
        worker_prefetch_cnt=config_dm.prefetch_factor,
    )

    print("----- Saving val batches -----")

    val_batch_pipe = _get_datapipe(
        config_dm.configuration,
        *config_dm.val_period,
        config.num_val_batches,
    )

    _save_batches_with_dataloader(
        batch_pipe=val_batch_pipe,
        batch_dir=f"{config.batch_output_dir}/val",
        num_batches=config.num_val_batches,
        rs_config=readingservice_config,
    )

    print("----- Saving train batches -----")

    train_batch_pipe = _get_datapipe(
        config_dm.configuration,
        *config_dm.train_period,
        config.num_train_batches,
    )

    _save_batches_with_dataloader(
        batch_pipe=train_batch_pipe,
        batch_dir=f"{config.batch_output_dir}/train",
        num_batches=config.num_train_batches,
        rs_config=readingservice_config,
    )

    print("done")


if __name__ == "__main__":
    main()
