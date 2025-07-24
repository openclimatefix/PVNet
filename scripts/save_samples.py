"""
Constructs samples and saves them to disk.

Currently a slightly hacky implementation due to the way the configs are done. This script will use
the same config file currently set to train the model.

use:
```
python save_samples.py
```
if setting all values in the datamodule config file, or

```
python save_samples.py \
    +datamodule.sample_output_dir="/mnt/disks/bigbatches/samples_v0" \
    +datamodule.num_train_samples=0 \
    +datamodule.num_val_samples=2 \
    datamodule.num_workers=2 \
    datamodule.prefetch_factor=2
```
if wanting to override these values for example
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

import dask
import hydra
from ocf_data_sampler.torch_datasets.datasets import PVNetUKRegionalDataset, SitesDataset
from ocf_data_sampler.torch_datasets.sample.site import SiteSample
from ocf_data_sampler.torch_datasets.sample.uk_regional import UKRegionalSample
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pvnet.utils import DATA_CONFIG_NAME, DATAMODULE_CONFIG_NAME, print_config

dask.config.set(scheduler="threads", num_workers=4)


# ------- filter warning and set up config  -------

logger = logging.getLogger(__name__)

logging.basicConfig(stream=sys.stdout, level=logging.ERROR)

# -------------------------------------------------


class SaveFuncFactory:
    """Factory for creating a function to save a sample to disk."""

    def __init__(self, save_dir: str, renewable: str = "pv_uk"):
        """Factory for creating a function to save a sample to disk."""
        self.save_dir = save_dir
        self.renewable = renewable

    def __call__(self, sample, sample_num: int):
        """Save a sample to disk"""
        save_path = f"{self.save_dir}/{sample_num:08}"

        if self.renewable == "pv_uk":
            sample_class = UKRegionalSample(sample)
            filename = f"{save_path}.pt"
        elif self.renewable == "site":
            sample_class = SiteSample(sample)
            filename = f"{save_path}.nc"
        else:
            raise ValueError(f"Unknown renewable: {self.renewable}")
        # Assign data and save
        sample_class._data = sample
        sample_class.save(filename)


def get_dataset(
    config_path: str, 
    start_time: str, 
    end_time: str, 
    renewable: str = "pv_uk",
) -> Dataset:
    """Get the dataset for the given renewable type."""
    if renewable == "pv_uk":
        dataset_cls = PVNetUKRegionalDataset
    elif renewable == "site":
        dataset_cls = SitesDataset
    else:
        raise ValueError(f"Unknown renewable: {renewable}")

    return dataset_cls(config_path, start_time=start_time, end_time=end_time)


def save_samples_with_dataloader(
    dataset: Dataset,
    save_dir: str,
    num_samples: int,
    dataloader_kwargs: dict,
    renewable: str = "pv_uk",
) -> None:
    """Save samples from a dataset using a dataloader."""
    save_func = SaveFuncFactory(save_dir, renewable=renewable)

    dataloader = DataLoader(dataset, **dataloader_kwargs)

    pbar = tqdm(total=num_samples)
    for i, sample in zip(range(num_samples), dataloader):
        save_func(sample, i)
        pbar.update()
    pbar.close()


@hydra.main(config_path="../configs/", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig) -> None:
    """Constructs and saves validation and training samples."""
    config_dm = config.datamodule

    print_config(config, resolve=False)

    # Set up directory
    os.makedirs(config_dm.sample_output_dir, exist_ok=False)

    # Copy across configs which define the samples into the new sample directory

    # Copy the datamodule config
    with open(f"{config_dm.sample_output_dir}/{DATAMODULE_CONFIG_NAME}", "w") as f:
        f.write(OmegaConf.to_yaml(config_dm))

    # Copy the data config
    shutil.copyfile(config_dm.configuration, f"{config_dm.sample_output_dir}/{DATA_CONFIG_NAME}")

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
        val_dataset = get_dataset(
            config_dm.configuration,
            *config_dm.val_period,
            renewable=config.renewable,
        )

        # Save samples
        save_samples_with_dataloader(
            dataset=val_dataset,
            save_dir=val_output_dir,
            num_samples=config_dm.num_val_samples,
            dataloader_kwargs=dataloader_kwargs,
            renewable=config.renewable,
        )

        del val_dataset

    if config_dm.num_train_samples > 0:
        print("----- Saving train samples -----")

        train_output_dir = f"{config_dm.sample_output_dir}/train"

        # Make directory for train samples
        os.mkdir(train_output_dir)

        # Get the dataset
        train_dataset = get_dataset(
            config_dm.configuration,
            *config_dm.train_period,
            renewable=config.renewable,
        )

        # Save samples
        save_samples_with_dataloader(
            dataset=train_dataset,
            save_dir=train_output_dir,
            num_samples=config_dm.num_train_samples,
            dataloader_kwargs=dataloader_kwargs,
            renewable=config.renewable,
        )

        del train_dataset

    print("----- Saving complete -----")


if __name__ == "__main__":
    main()
