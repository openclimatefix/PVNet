""" Data module for pytorch lightning """
from typing import Callable, Union

import fsspec.asyn
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, default_collate

from ocf_datapipes.batch.fake.fake_batch import fake_data_pipeline
from ocf_datapipes.training.pvnet import pvnet_datapipe

from ocf_datapipes.utils.utils import (
    stack_np_examples_into_batch,
    set_fsspec_for_multiprocess,
)

from datetime import datetime


def worker_init_fn(worker_id):
    """Configures each dataset worker process.
    1. Get fsspec ready for multi process
    """
    # fix for fsspec when using multprocess
    set_fsspec_for_multiprocess()
    

class DataModule(LightningDataModule):
    """
    Example of LightningDataModule using ocf_datapipes
    """

    def __init__(
        self,
        configuration,
        fake_data,
        n_train_batches,
        n_val_batches,
        n_test_batches,
        batch_size=16,
        num_workers=0,
        prefetch_factor=2,
        train_period=[None, None],
        val_period=[None, None],
        test_period=[None, None],
    ):

        super().__init__()
        self.configuration = configuration
        self.fake_data = fake_data
        self.n_train_batches = n_train_batches
        self.n_val_batches = n_val_batches
        self.n_test_batches = n_test_batches
        self.batch_size = batch_size

        self.train_period = [
            None if d is None else datetime.strptime(d, "%Y-%m-%d") 
            for d in train_period
        ]
        self.val_period = [
            None if d is None else datetime.strptime(d, "%Y-%m-%d") 
            for d in val_period
        ]
        self.test_period = [
            None if d is None else datetime.strptime(d, "%Y-%m-%d") 
            for d in test_period
        ]

        self.dataloader_config = dict(
            pin_memory=True,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            worker_init_fn=worker_init_fn,
            # Persistent_workers option needs num_workers > 0
            persistent_workers=num_workers > 0,
            # Disable automatic batching because dataset
            # returns complete batches.
            batch_size=None,
        )
        
    def _get_datapipe(self, start_time, end_time, n_batches, batch_size):
        if self.fake_data:
            data_pipeline = fake_data_pipeline(
                configuration=self.configuration
            )
        else:
            data_pipeline = pvnet_datapipe(
                self.configuration, 
                start_time=start_time,
                end_time=end_time,
            )
        
        data_pipeline = (
            data_pipeline
                .batch(batch_size)
                .map(stack_np_examples_into_batch)
        )
        
        if n_batches is not None:
            data_pipeline = data_pipeline.set_length(n_batches)
            
        return data_pipeline
        
    def train_dataloader(self):
        datapipe = self._get_datapipe(*self.train_period, self.n_train_batches, self.batch_size)
        return DataLoader(datapipe, **self.dataloader_config)

    def val_dataloader(self):
        datapipe = self._get_datapipe(*self.val_period, self.n_val_batches, self.batch_size)
        kwargs = dict(**self.dataloader_config)
        kwargs['num_workers'] = 0
        kwargs['prefetch_factor'] = 2
        kwargs['worker_init_fn'] = None
        kwargs['persistent_workers'] = False
        return DataLoader(datapipe, **kwargs)

    def test_dataloader(self):
        datapipe = self._get_datapipe(*self.test_period, self.n_test_batches, self.batch_size)
        kwargs = dict(**self.dataloader_config)
        kwargs['num_workers'] = 0
        kwargs['prefetch_factor'] = 2
        kwargs['worker_init_fn'] = None
        kwargs['persistent_workers'] = False
        return DataLoader(datapipe, **kwargs)

