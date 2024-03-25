"""
Constructs batches and saves them to disk.

Currently a slightly hacky implementation due to the way the configs are done. This script will use
the same config file currently set to train the model.

use:
```
python save_batches.py \
    +batch_output_dir="/mnt/disks/bigbatches/batches_v0" \
    datamodule.batch_size=2 \
    datamodule.num_workers=2 \
    +num_train_batches=0 \
    +num_val_batches=2
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

# Tired of seeing these warnings
import warnings

import hydra
import torch
from ocf_datapipes.batch import stack_np_examples_into_batch
from ocf_datapipes.training.pvnet import pvnet_datapipe
from ocf_datapipes.training.pvnet_site import pvnet_site_datapipe
from ocf_datapipes.training.windnet import windnet_datapipe
from omegaconf import DictConfig, OmegaConf
from sqlalchemy import exc as sa_exc
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter import IterableWrapper
from tqdm import tqdm

from pvnet.data.utils import batch_to_tensor
from pvnet.utils import print_config

warnings.filterwarnings("ignore", category=sa_exc.SAWarning)

logger = logging.getLogger(__name__)

logging.basicConfig(stream=sys.stdout, level=logging.ERROR)


class _save_batch_func_factory:
    def __init__(self, batch_dir, output_format: str = "torch"):
        self.batch_dir = batch_dir
        self.output_format = output_format

    def __call__(self, input):
        i, batch = input
        if self.output_format == "torch":
            torch.save(batch, f"{self.batch_dir}/{i:06}.pt")
        elif self.output_format == "netcdf":
            batch.to_netcdf(f"{self.batch_dir}/{i:06}.nc", mode="w", engine="h5netcdf")


def _get_datapipe(config_path, start_time, end_time, batch_size, renewable: str = "pv"):
    if renewable == "pv":
        data_pipeline_fn = pvnet_datapipe
    elif renewable == "wind":
        data_pipeline_fn = windnet_datapipe
    elif renewable == "pv_india":
        data_pipeline_fn = pvnet_site_datapipe
    else:
        raise ValueError(f"Unknown renewable: {renewable}")
    data_pipeline = data_pipeline_fn(
        config_path,
        start_time=start_time,
        end_time=end_time,
    )
    if renewable == "pv":
        data_pipeline = (
            data_pipeline.batch(batch_size).map(stack_np_examples_into_batch).map(batch_to_tensor)
        )
    return data_pipeline


def _save_batches_with_dataloader(
    batch_pipe, batch_dir, num_batches, dataloader_kwargs, output_format: str = "torch"
):
    save_func = _save_batch_func_factory(batch_dir, output_format=output_format)
    filenumber_pipe = IterableWrapper(range(num_batches)).sharding_filter()
    save_pipe = filenumber_pipe.zip(batch_pipe).map(save_func)

    dataloader = DataLoader(save_pipe, **dataloader_kwargs)

    pbar = tqdm(total=num_batches)
    for i, batch in zip(range(num_batches), dataloader):
        pbar.update()
    pbar.close()
    del dataloader


@hydra.main(config_path="../configs/", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig):
    """Constructs and saves validation and training batches."""
    config_dm = config.datamodule

    print_config(config, resolve=False)

    # Set up directory
    os.makedirs(config.batch_output_dir, exist_ok=False)

    with open(f"{config.batch_output_dir}/datamodule.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(config.datamodule))

    shutil.copyfile(config_dm.configuration, f"{config.batch_output_dir}/data_configuration.yaml")

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

    if config.num_val_batches > 0:
        os.mkdir(f"{config.batch_output_dir}/val")
        print("----- Saving val batches -----")

        val_batch_pipe = _get_datapipe(
            config_dm.configuration,
            *config_dm.val_period,
            config_dm.batch_size,
            renewable=config.renewable,
        )

        _save_batches_with_dataloader(
            batch_pipe=val_batch_pipe,
            batch_dir=f"{config.batch_output_dir}/val",
            num_batches=config.num_val_batches,
            dataloader_kwargs=dataloader_kwargs,
            output_format="torch" if config.renewable == "pv" else "netcdf",
        )

    if config.num_train_batches > 0:
        os.mkdir(f"{config.batch_output_dir}/train")
        print("----- Saving train batches -----")

        train_batch_pipe = _get_datapipe(
            config_dm.configuration,
            *config_dm.train_period,
            config_dm.batch_size,
            renewable=config.renewable,
        )

        _save_batches_with_dataloader(
            batch_pipe=train_batch_pipe,
            batch_dir=f"{config.batch_output_dir}/train",
            num_batches=config.num_train_batches,
            dataloader_kwargs=dataloader_kwargs,
            output_format="torch" if config.renewable == "pv" else "netcdf",
        )

    print("done")


if __name__ == "__main__":
    main()
