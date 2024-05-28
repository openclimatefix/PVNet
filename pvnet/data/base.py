""" Data module for pytorch lightning """
from datetime import datetime

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader


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
        shuffle_factor=100,
        nwp_channels=None,
    ):
        """Datamodule for training pvnet architecture.

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
            shuffle_factor: Number of presaved batches to be split and reshuffled to create returned
                batches. A larger factor means on each epoch the batches will be more diverse but at
                the cost of using more RAM.
            nwp_channels: Number of NWP channels to use. If None, the all channels are used
        """
        super().__init__()
        self.configuration = configuration
        self.batch_size = batch_size
        self.batch_dir = batch_dir
        self.shuffle_factor = shuffle_factor
        self.nwp_channels = nwp_channels

        if not ((batch_dir is not None) ^ (configuration is not None)):
            raise ValueError("Exactly one of `batch_dir` or `configuration` must be set.")

        if (nwp_channels is not None) and (batch_dir is None):
            raise ValueError(
                "In order for 'nwp_channels' to work, we need batch_dir. "
                "Otherwise the nwp channels is one in the configuration"
            )

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
        raise NotImplementedError

    def _get_premade_batches_datapipe(self, subdir, shuffle=False):
        raise NotImplementedError

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
