""" Data module for pytorch lightning """
from glob import glob

import torch
from lightning.pytorch import LightningDataModule
from ocf_data_sampler.torch_datasets import PVNetUKRegionalDataset, SitesDataset
from ocf_datapipes.batch import (
    NumpyBatch,
    TensorBatch,
    batch_to_tensor,
    stack_np_examples_into_batch,
)
from torch.utils.data import DataLoader, Dataset


class NumpybatchPremadeSamplesDataset(Dataset):
    """Dataset to load NumpyBatch samples"""

    def __init__(self, sample_dir):
        """Dataset to load NumpyBatch samples

        Args:
            sample_dir: Path to the directory of pre-saved samples.
        """
        self.sample_paths = glob(f"{sample_dir}/*.pt")

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        return torch.load(self.sample_paths[idx])


def collate_fn(samples: list[NumpyBatch]) -> TensorBatch:
    """Convert a list of NumpyBatch samples to a tensor batch"""
    return batch_to_tensor(stack_np_examples_into_batch(samples))


class DataModule(LightningDataModule):
    """Datamodule for training pvnet and using pvnet pipeline in `ocf_datapipes`."""

    def __init__(
        self,
        configuration: str | None = None,
        sample_dir: str | None = None,
        batch_size: int = 16,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        train_period: list[str | None] = [None, None],
        val_period: list[str | None] = [None, None],
    ):
        """Datamodule for training pvnet architecture.

        Can also be used with pre-made batches if `sample_dir` is set.

        Args:
            configuration: Path to datapipe configuration file.
            sample_dir: Path to the directory of pre-saved samples. Cannot be used together with
                `configuration` or '[train/val]_period'.
            batch_size: Batch size.
            num_workers: Number of workers to use in multiprocess batch loading.
            prefetch_factor: Number of data will be prefetched at the end of each worker process.
            train_period: Date range filter for train dataloader.
            val_period: Date range filter for val dataloader.

        """
        super().__init__()

        if not ((sample_dir is not None) ^ (configuration is not None)):
            raise ValueError("Exactly one of `sample_dir` or `configuration` must be set.")

        if sample_dir is not None:
            if any([period != [None, None] for period in [train_period, val_period]]):
                raise ValueError("Cannot set `(train/val)_period` with presaved samples")

        self.configuration = configuration
        self.sample_dir = sample_dir
        self.train_period = train_period
        self.val_period = val_period

        self._common_dataloader_kwargs = dict(
            batch_size=batch_size,
            sampler=None,
            batch_sampler=None,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
            prefetch_factor=prefetch_factor,
            persistent_workers=False,
        )

    def _get_streamed_samples_dataset(self, start_time, end_time) -> Dataset:
        if self.configuration.renewable == "pv":
            return PVNetUKRegionalDataset(
                self.configuration, start_time=start_time, end_time=end_time
            )
        elif self.configuration.renewable in ["wind", "pv_india", "pv_site"]:
            return SitesDataset(self.configuration, start_time=start_time, end_time=end_time)
        else:
            raise ValueError(f"Unknown renewable: {self.configuration.renewable}")

    def _get_premade_samples_dataset(self, subdir) -> Dataset:
        split_dir = f"{self.sample_dir}/{subdir}"
        return NumpybatchPremadeSamplesDataset(split_dir)

    def train_dataloader(self) -> DataLoader:
        """Construct train dataloader"""
        if self.sample_dir is not None:
            dataset = self._get_premade_samples_dataset("train")
        else:
            dataset = self._get_streamed_samples_dataset(*self.train_period)
        return DataLoader(dataset, shuffle=True, **self._common_dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        """Construct val dataloader"""
        if self.sample_dir is not None:
            dataset = self._get_premade_samples_dataset("val")
        else:
            dataset = self._get_streamed_samples_dataset(*self.val_period)
        return DataLoader(dataset, shuffle=False, **self._common_dataloader_kwargs)
