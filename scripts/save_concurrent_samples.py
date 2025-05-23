"""
Constructs batches where each batch includes all GSPs and only a single timestamp.

Currently a slightly hacky implementation due to the way the configs are done. This script will use
the same config file currently set to train the model. In the datamodule config it is possible
to set the batch_output_dir and number of train/val batches, they can also be overriden in the
command as shown in the example below.

use:
```
python save_concurrent_samples.py \
    +datamodule.sample_output_dir="/mnt/disks/concurrent_batches/concurrent_samples_sat_pred_test" \
    +datamodule.num_train_samples=20 \
    +datamodule.num_val_samples=20
```

"""
# Ensure this block of code runs only in the main process to avoid issues with worker processes.
if __name__ == "__main__":
    import torch.multiprocessing as mp

    # Set the start method for torch multiprocessing. Choose either "forkserver" or "spawn" to be
    # compatible with dask's multiprocessing.
    mp.set_start_method("forkserver")

    # Set the sharing strategy to 'file_system' to handle file descriptor limitations. This is
    # important because libraries like Zarr may open many files, which can exhaust the file
    # descriptor limit if too many workers are used.
    mp.set_sharing_strategy("file_system")


import logging
import os
import shutil
import sys
import warnings

import hydra
import numpy as np
import torch
from ocf_data_sampler.torch_datasets.datasets.pvnet_uk import PVNetUKConcurrentDataset
from omegaconf import DictConfig, OmegaConf
from sqlalchemy import exc as sa_exc
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pvnet.utils import print_config

# ------- filter warning and set up config  -------

warnings.filterwarnings("ignore", category=sa_exc.SAWarning)

logger = logging.getLogger(__name__)

logging.basicConfig(stream=sys.stdout, level=logging.ERROR)

# -------------------------------------------------


class SaveFuncFactory:
    """Factory for creating a function to save a sample to disk."""

    def __init__(self, save_dir: str):
        """Factory for creating a function to save a sample to disk."""
        self.save_dir = save_dir

    def __call__(self, sample, sample_num: int):
        """Save a sample to disk"""
        torch.save(sample, f"{self.save_dir}/{sample_num:08}.pt")


def save_samples_with_dataloader(
    dataset: Dataset,
    save_dir: str,
    num_samples: int,
    dataloader_kwargs: dict,
) -> None:
    """Save samples from a dataset using a dataloader."""
    save_func = SaveFuncFactory(save_dir)

    gsp_ids = np.array([loc.id for loc in dataset.locations])

    dataloader = DataLoader(dataset, **dataloader_kwargs)

    pbar = tqdm(total=num_samples)
    for i, sample in zip(range(num_samples), dataloader):
        check_sample(sample, gsp_ids)
        save_func(sample, i)
        pbar.update()
    pbar.close()


def check_sample(sample, gsp_ids):
    """Check if sample is valid concurrent batch for all GSPs"""
    # Check all GSP IDs are included and in correct order
    assert (sample["gsp_id"].flatten().numpy() == gsp_ids).all()
    # Check all times are the same
    assert len(np.unique(sample["gsp_time_utc"][:, 0].numpy())) == 1


@hydra.main(config_path="../configs/", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig) -> None:
    """Constructs and saves validation and training samples."""
    config_dm = config.datamodule

    print_config(config, resolve=False)

    # Set up directory
    os.makedirs(config_dm.sample_output_dir, exist_ok=False)

    # Copy across configs which define the samples into the new sample directory
    with open(f"{config_dm.sample_output_dir}/datamodule.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(config_dm))

    shutil.copyfile(
        config_dm.configuration, f"{config_dm.sample_output_dir}/data_configuration.yaml"
    )

    # Define the keywargs going into the train and val dataloaders
    dataloader_kwargs = dict(
        shuffle=True,
        batch_size=None,
        sampler=None,
        batch_sampler=None,
        num_workers=config_dm.num_workers,
        collate_fn=None,
        pin_memory=False,  # Only using CPU to prepare samples so pinning is not beneficial
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        prefetch_factor=config_dm.prefetch_factor,
        persistent_workers=False,  # Not needed since we only enter the dataloader loop once
    )

    if config_dm.num_val_samples > 0:
        print("----- Saving val samples -----")

        val_output_dir = f"{config_dm.sample_output_dir}/val"

        # Make directory for val samples
        os.mkdir(val_output_dir)

        # Get the dataset
        val_dataset = PVNetUKConcurrentDataset(
            config_dm.configuration,
            start_time=config_dm.val_period[0],
            end_time=config_dm.val_period[1],
        )

        # Save samples
        save_samples_with_dataloader(
            dataset=val_dataset,
            save_dir=val_output_dir,
            num_samples=config_dm.num_val_samples,
            dataloader_kwargs=dataloader_kwargs,
        )

        del val_dataset

    if config_dm.num_train_samples > 0:
        print("----- Saving train samples -----")

        train_output_dir = f"{config_dm.sample_output_dir}/train"

        # Make directory for train samples
        os.mkdir(train_output_dir)

        # Get the dataset
        train_dataset = PVNetUKConcurrentDataset(
            config_dm.configuration,
            start_time=config_dm.train_period[0],
            end_time=config_dm.train_period[1],
        )

        # Save samples
        save_samples_with_dataloader(
            dataset=train_dataset,
            save_dir=train_output_dir,
            num_samples=config_dm.num_train_samples,
            dataloader_kwargs=dataloader_kwargs,
        )

        del train_dataset

    print("----- Saving complete -----")


if __name__ == "__main__":
    main()
