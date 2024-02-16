""" Data module for pytorch lightning """
from datetime import datetime

import torch
from lightning.pytorch import LightningDataModule
from ocf_datapipes.batch import stack_np_examples_into_batch
from ocf_datapipes.training.pvnet import pvnet_datapipe
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter import FileLister

from pvnet.data.utils import batch_to_tensor


class BaseDataModule(LightningDataModule):
    """Datamodule for training pvnet and using pvnet pipeline in `ocf_datapipes`."""

    def __init__(
        self,
        configuration=None,
        batch_size=16,
        num_workers=0,
        prefetch_factor=None,
        train_period=[None, None],
        val_period=[None, None],
        test_period=[None, None],
        batch_dir=None,
    ):
        """Datamodule for training pvnet and using pvnet pipeline in `ocf_datapipes`.

        Can also be used with pre-made batches if `batch_dir` is set.


        Args:
            configuration: Path to datapipe configuration file.
            batch_size: Batch size.
            num_workers: Number of workers to use in multiprocess batch loading.
            prefetch_factor: Number of data will be prefetched at the end of each worker process.
            train_period: Date range filter for train dataloader.
            val_period: Date range filter for val dataloader.
            test_period: Date range filter for test dataloader.
            batch_dir: Path to the directory of pre-saved batches. Cannot be used together with
                `configuration` or 'train/val/test_period'.

        """
        super().__init__()
        self.configuration = configuration
        self.batch_size = batch_size
        self.batch_dir = batch_dir

        if not ((batch_dir is not None) ^ (configuration is not None)):
            raise ValueError("Exactly one of `batch_dir` or `configuration` must be set.")

        if batch_dir is not None:
            if any([period != [None, None] for period in [train_period, val_period, test_period]]):
                raise ValueError("Cannot set `(train/val/test)_period` with presaved batches")

        self.train_period = [
            None if d is None else datetime.strptime(d, "%Y-%m-%d") for d in train_period
        ]
        self.val_period = [
            None if d is None else datetime.strptime(d, "%Y-%m-%d") for d in val_period
        ]
        self.test_period = [
            None if d is None else datetime.strptime(d, "%Y-%m-%d") for d in test_period
        ]

        self._common_dataloader_kwargs = dict(
            batch_size=None,  # batched in datapipe step
            sampler=None,
            batch_sampler=None,
            num_workers=num_workers,
            collate_fn=None,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
            prefetch_factor=prefetch_factor,
            persistent_workers=False,
        )

    def _get_datapipe(self, start_time, end_time):
        data_pipeline = pvnet_datapipe(
            self.configuration,
            start_time=start_time,
            end_time=end_time,
        )

        data_pipeline = (
            data_pipeline.batch(self.batch_size)
            .map(stack_np_examples_into_batch)
            .map(batch_to_tensor)
        )
        return data_pipeline

    def _get_premade_batches_datapipe(self, subdir, shuffle=False):
        data_pipeline = FileLister(f"{self.batch_dir}/{subdir}", masks="*.pt", recursive=False)
        if shuffle:
            data_pipeline = (
                data_pipeline.shuffle(buffer_size=10_000)
                .sharding_filter()
                .map(torch.load)
                # Split the batches and reshuffle them to be combined into new batches
                .split_batches()
                .shuffle(buffer_size=100 * self.batch_size)
            )
        else:
            data_pipeline = (
                data_pipeline.sharding_filter().map(torch.load)
                # Split the batches so we can use any batch-size
                .split_batches()
            )

        data_pipeline = (
            data_pipeline.batch(self.batch_size)
            .map(stack_np_examples_into_batch)
            .map(batch_to_tensor)
        )

        return data_pipeline

    def train_dataloader(self):
        """Construct train dataloader"""
        if self.batch_dir is not None:
            datapipe = self._get_premade_batches_datapipe("train", shuffle=True)
        else:
            datapipe = self._get_datapipe(*self.train_period)
        return DataLoader(datapipe, shuffle=True, **self._common_dataloader_kwargs)

    def val_dataloader(self):
        """Construct val dataloader"""
        if self.batch_dir is not None:
            datapipe = self._get_premade_batches_datapipe("val")
        else:
            datapipe = self._get_datapipe(*self.val_period)
        return DataLoader(datapipe, shuffle=False, **self._common_dataloader_kwargs)

    def test_dataloader(self):
        """Construct test dataloader"""
        if self.batch_dir is not None:
            datapipe = self._get_premade_batches_datapipe("test")
        else:
            datapipe = self._get_datapipe(*self.test_period)
        return DataLoader(datapipe, shuffle=False, **self._common_dataloader_kwargs)
