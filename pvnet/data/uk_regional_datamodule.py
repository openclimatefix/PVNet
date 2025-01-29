""" Data module for pytorch lightning """
from glob import glob

import torch
from ocf_data_sampler.torch_datasets.datasets.pvnet_uk_regional import PVNetUKRegionalDataset
from ocf_data_sampler.sample.uk_regional import UKRegionalSample 
from torch.utils.data import Dataset

from pvnet.data.base_datamodule import BaseDataModule


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
        sample = UKRegionalSample.load(self.sample_paths[idx])        
        return sample.to_numpy()


class DataModule(BaseDataModule):
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
        super().__init__(
            configuration=configuration,
            sample_dir=sample_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            train_period=train_period,
            val_period=val_period,
        )

    def _get_streamed_samples_dataset(self, start_time, end_time) -> Dataset:
        return PVNetUKRegionalDataset(self.configuration, start_time=start_time, end_time=end_time)

    def _get_premade_samples_dataset(self, subdir) -> Dataset:
        split_dir = f"{self.sample_dir}/{subdir}"
        return NumpybatchPremadeSamplesDataset(split_dir)
