""" Data module for pytorch lightning """
from typing import Callable, Union
from glob import glob
from datetime import datetime

import numpy as np
import torch
from torchdata.datapipes.iter import FileLister
from torchdata.datapipes import functional_datapipe
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from lightning.pytorch import LightningDataModule

from ocf_datapipes.training.pvnet import pvnet_datapipe
from ocf_datapipes.utils.utils import stack_np_examples_into_batch
from ocf_datapipes.utils.consts import BatchKey


    
def batch_to_tensor(batch):
    for k in list(batch.keys()):
        if isinstance(batch[k], np.ndarray) and np.issubdtype(batch[k].dtype, np.number):
            batch[k] = torch.as_tensor(batch[k])    
    return batch

def print_yaml(path):
    print(f"{path} :")
    with open(path, mode="r") as stream:
        print("".join(stream.readlines()))
    
    
from torchdata.datapipes.iter import IterDataPipe



def split_batches(batch):
    n_samples = batch[BatchKey.gsp].shape[0]
    keys = list(batch.keys())
    examples = [{} for _ in range(n_samples)]
    for i in range(n_samples):
        b = examples[i]
        for k in keys:
            if ("idx" in k.name) or ("channel_names" in k.name):
                b[k] = batch[k]
            else:
                b[k] = batch[k][i]
    return examples

@functional_datapipe("split_batches")
class BatchSplitter(IterDataPipe):

    def __init__(self, source_datapipe: IterDataPipe):
        """

        """
        self.source_datapipe = source_datapipe

    def __iter__(self):
        """Opens the NWP data"""
        for batch in self.source_datapipe:
            for example in split_batches(batch):
                yield example


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
        use_premade_batches=False,
        batch_dir=None,
    ):

        super().__init__()
        self.configuration = configuration
        self.batch_size = batch_size
        self.block_nwp_and_sat = block_nwp_and_sat
        self.use_premade_batches = use_premade_batches
        self.batch_dir = batch_dir
        
        if use_premade_batches:
            print(
                f"Loading batches from: {batch_dir}\n"
                "These batches were saved with the following configs:"
            )
            print_yaml(f"{batch_dir}/datamodule.yaml")
            print_yaml(f"{batch_dir}/data_configuration.yaml")

            n_train = len(glob(f"{batch_dir}/train/*.pt"))
            n_val = len(glob(f"{batch_dir}/val/*.pt"))
            
            print(f"There are {n_train} training batches")
            print(f"There are {n_val} validation batches")

                    
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
        )
        
    def _get_datapipe(self, start_time, end_time):
        data_pipeline = pvnet_datapipe(
            self.configuration, 
            start_time=start_time,
            end_time=end_time,
            block_sat=self.block_nwp_and_sat,
            block_nwp=self.block_nwp_and_sat,
        )

        data_pipeline = (
            data_pipeline
                .batch(self.batch_size)
                .map(stack_np_examples_into_batch)
                .map(batch_to_tensor)
        )   
        return data_pipeline
    
    def _get_premade_batches_datapipe(self, subdir, shuffle=False):
        data_pipeline = FileLister(f"{self.batch_dir}/{subdir}", masks="*.pt", recursive=False)
        if shuffle:
            data_pipeline = (
                data_pipeline
                .shuffle(buffer_size=10_000)
                .sharding_filter()
                .map(torch.load)
                # Split the batches and reshuffle them to be combined into new batches
                .split_batches()
                .shuffle(buffer_size=100*self.batch_size)
            )
        else:
            data_pipeline = (
                data_pipeline
                .sharding_filter()
                .map(torch.load)
                # Split the batches so we can use any batch-size
                .split_batches()
            )
        
        data_pipeline = (
            data_pipeline
            .batch(self.batch_size)
            .map(stack_np_examples_into_batch)
            .map(batch_to_tensor)
        )
        
        return data_pipeline
        
    def train_dataloader(self):
        if self.use_premade_batches:
            datapipe = self._get_premade_batches_datapipe("train", shuffle=True)
        else:
            datapipe = self._get_datapipe(*self.train_period)
        rs = MultiProcessingReadingService(**self.readingservice_config)
        return DataLoader2(datapipe, reading_service=rs)
        
    def val_dataloader(self):
        if self.use_premade_batches:
            datapipe = self._get_premade_batches_datapipe("val")
        else:
            datapipe = self._get_datapipe(*self.val_period)
        rs = MultiProcessingReadingService(**self.readingservice_config)
        return DataLoader2(datapipe, reading_service=rs)
        

    def test_dataloader(self):
        if self.use_premade_batches:
            datapipe = self._get_premade_batches_datapipe("test")
        else:
            datapipe = self._get_datapipe(*self.test_period)
        rs = MultiProcessingReadingService(**self.readingservice_config)
        return DataLoader2(datapipe, reading_service=rs)
        

