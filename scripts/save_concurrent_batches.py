"""
Constructs batches where each batch includes all GSPs and only a single timestamp.

Currently a slightly hacky implementation due to the way the configs are done. This script will use
the same config file currently set to train the model.

use:
```
python save_concurrent_batches.py \
    +batch_output_dir="/mnt/disks/nwp_rechunk/concurrent_batches_v3.9" \
    +num_train_batches=20_000 \
    +num_val_batches=4_000
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
from ocf_datapipes.batch import stack_np_examples_into_batch
from ocf_datapipes.training.common import (
    open_and_return_datapipes,
)
from ocf_datapipes.training.pvnet import construct_loctime_pipelines, construct_sliced_data_pipeline
from ocf_datapipes.utils.consts import BatchKey
from omegaconf import DictConfig, OmegaConf
from sqlalchemy import exc as sa_exc
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter import IterableWrapper
from tqdm import tqdm

from pvnet.data.utils import batch_to_tensor
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


def select_first(x):
    """Select zeroth element from indexable object"""
    return x[0]


def _get_loctimes_datapipes(config_path, start_time, end_time, n_batches):
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

    # These two datapipes come from an earlier fork and must be iterated through together
    # despite the fact that we don't want these random locations here
    random_location_datapipe, t0_datapipe = construct_loctime_pipelines(
        config_path,
        start_time,
        end_time,
    )

    # Iterate through both but select only time
    t0_datapipe = t0_datapipe.zip(random_location_datapipe).map(select_first)

    # Create times datapipe so we'll get the same time over each batch
    t0_datapipe = t0_datapipe.header(n_batches)
    t0_datapipe = IterableWrapper([[t0 for gsp_id in range(1, 318)] for t0 in t0_datapipe])
    t0_datapipe = t0_datapipe.sharding_filter()
    t0_datapipe = t0_datapipe.unbatch(unbatch_level=1)

    return location_pipe, t0_datapipe


def _get_datapipe(config_path, start_time, end_time, n_batches):
    # Open datasets from the config and filter to useable location-time pairs

    location_pipe, t0_datapipe = _get_loctimes_datapipes(
        config_path, start_time, end_time, n_batches
    )

    data_pipeline = construct_sliced_data_pipeline(
        config_path,
        location_pipe,
        t0_datapipe,
    )

    data_pipeline = data_pipeline.batch(317).map(stack_np_examples_into_batch).map(batch_to_tensor)

    return data_pipeline


def _save_batches_with_dataloader(batch_pipe, batch_dir, num_batches, dataloader_kwargs):
    save_func = _save_batch_func_factory(batch_dir)
    filenumber_pipe = IterableWrapper(np.arange(num_batches)).sharding_filter()
    save_pipe = filenumber_pipe.zip(batch_pipe).map(save_func)

    dataloader = DataLoader(save_pipe, **dataloader_kwargs)

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

    dataloader_kwargs = dict(
        shuffle=False,
        batch_size=None,  # batched in datapipe step
        sampler=None,
        batch_sampler=None,
        num_workers=config_dm.num_workers,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        prefetch_factor=config_dm.prefetch_factor,
        persistent_workers=False,
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
        dataloader_kwargs=dataloader_kwargs,
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
        dataloader_kwargs=dataloader_kwargs,
    )

    print("done")


if __name__ == "__main__":
    main()
