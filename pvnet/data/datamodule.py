""" Data module for pytorch lightning """
from typing import Callable, Union

import fsspec.asyn
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, default_collate
import numpy as np
import torch

from ocf_datapipes.batch.fake.fake_batch import fake_data_pipeline
from ocf_datapipes.training.pvnet import pvnet_datapipe

from ocf_datapipes.utils.utils import (
    stack_np_examples_into_batch,
    set_fsspec_for_multiprocess,
)

from datetime import datetime

from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService


def worker_init_fn(*args, **kwargs):
    """Configures each dataset worker process.
    1. Get fsspec ready for multi process
    """
    # fix for fsspec when using multprocess
    set_fsspec_for_multiprocess()
    
def batch_to_tensor(batch):
    for k in list(batch.keys()):
        if isinstance(batch[k], np.ndarray) and np.issubdtype(batch[k].dtype, np.number):
            batch[k] = torch.as_tensor(batch[k])    
    return batch
    

class DataModule(LightningDataModule):
    """
    Example of LightningDataModule using ocf_datapipes
    """

    def __init__(
        self,
        configuration,
        batch_size=16,
        num_workers=0,
        prefetch_factor=2,
        train_period=[None, None],
        val_period=[None, None],
        test_period=[None, None],
        block_nwp_and_sat=False,
    ):

        super().__init__()
        self.configuration = configuration
        self.batch_size = batch_size
        self.block_nwp_and_sat = block_nwp_and_sat

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

        self.readingservice_config = dict(
            num_workers=num_workers,
            multiprocessing_context="spawn",
            worker_prefetch_cnt=prefetch_factor,
            #worker_init_fn=worker_init_fn,
        )
        self.dataloader_config = dict(
            pin_memory=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            worker_init_fn=worker_init_fn,
            persistent_workers=False,
            # Disable automatic batching because dataset
            # returns complete batches.
            batch_size=None,
        )
        
    def _get_datapipe(self, start_time, end_time, batch_size):

        data_pipeline = pvnet_datapipe(
            self.configuration, 
            start_time=start_time,
            end_time=end_time,
            block_sat=self.block_nwp_and_sat,
            block_nwp=self.block_nwp_and_sat,
        )

        data_pipeline = (
            data_pipeline
                .batch(batch_size)
                .map(stack_np_examples_into_batch)
                .map(batch_to_tensor)
        )   
        return data_pipeline
        
    def train_dataloader(self):
        datapipe = self._get_datapipe(*self.train_period, self.batch_size)
        #rs = MultiProcessingReadingService(**self.readingservice_config)
        #return DataLoader2(datapipe, reading_service=rs)
        kwargs = dict(**self.dataloader_config)
        return DataLoader(datapipe, **kwargs)
        
    def val_dataloader(self):
        datapipe = self._get_datapipe(*self.val_period, self.batch_size)
        #rs = MultiProcessingReadingService(**self.readingservice_config)
        #return DataLoader2(datapipe, reading_service=rs)
        kwargs = dict(**self.dataloader_config)
        return DataLoader(datapipe, **kwargs)
        

    def test_dataloader(self):
        datapipe = self._get_datapipe(*self.test_period, self.batch_size)
        #rs = MultiProcessingReadingService(**self.readingservice_config)
        #return DataLoader2(datapipe, reading_service=rs)
        kwargs = dict(**self.dataloader_config)
        return DataLoader(datapipe, **kwargs)
        

