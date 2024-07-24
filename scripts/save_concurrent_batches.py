"""
Constructs batches where each batch includes all GSPs and only a single timestamp.

Currently a slightly hacky implementation due to the way the configs are done. This script will use
the same config file currently set to train the model. In the datamodule config it is possible
to set the batch_output_dir and number of train/val batches, they can also be overriden in the
command as shown in the example below.

use:
```
python save_concurrent_batches.py \
    datamodule.batch_output_dir="/mnt/disks/nwp_rechunk/concurrent_batches_v3.9" \
    datamodule.num_train_batches=20_000 \
    datamodule.num_val_batches=4_000
```

"""
# This is needed to get multiprocessing/multiple workers to behave
try:
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import logging
import os
import shutil
import sys
import warnings

import hydra
import numpy as np
import torch
from ocf_datapipes.batch import BatchKey, batch_to_tensor, stack_np_examples_into_batch
from ocf_datapipes.training.common import (
    open_and_return_datapipes,
)
from ocf_datapipes.training.pvnet_all_gsp import (
    construct_time_pipeline, construct_sliced_data_pipeline
)
from omegaconf import DictConfig, OmegaConf
from sqlalchemy import exc as sa_exc
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter import IterableWrapper
from tqdm import tqdm


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

    t0_datapipe = construct_time_pipeline(
        config_path,
        start_time,
        end_time,
    )

    t0_datapipe = t0_datapipe.header(n_batches)
    t0_datapipe = t0_datapipe.sharding_filter()
    
    datapipe = construct_sliced_data_pipeline(
        config_path,
        t0_datapipe,
    )

    return datapipe


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
    os.makedirs(config_dm.batch_output_dir, exist_ok=False)

    with open(f"{config_dm.batch_output_dir}/datamodule.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(config.datamodule))

    shutil.copyfile(
        config_dm.configuration, f"{config_dm.batch_output_dir}/data_configuration.yaml"
    )

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

    if config_dm.num_val_batches > 0:
        print("----- Saving val batches -----")

        os.mkdir(f"{config_dm.batch_output_dir}/val")

        val_batch_pipe = _get_datapipe(
            config_dm.configuration,
            *config_dm.val_period,
            config_dm.num_val_batches,
        )

        _save_batches_with_dataloader(
            batch_pipe=val_batch_pipe,
            batch_dir=f"{config_dm.batch_output_dir}/val",
            num_batches=config_dm.num_val_batches,
            dataloader_kwargs=dataloader_kwargs,
        )

    if config_dm.num_train_batches > 0:
        print("----- Saving train batches -----")

        os.mkdir(f"{config_dm.batch_output_dir}/train")

        train_batch_pipe = _get_datapipe(
            config_dm.configuration,
            *config_dm.train_period,
            config_dm.num_train_batches,
        )

        _save_batches_with_dataloader(
            batch_pipe=train_batch_pipe,
            batch_dir=f"{config_dm.batch_output_dir}/train",
            num_batches=config_dm.num_train_batches,
            dataloader_kwargs=dataloader_kwargs,
        )

    print("done")


if __name__ == "__main__":
    main()
