""" Data module for pytorch lightning """
from typing import Callable, Union

import fsspec.asyn
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, default_collate

from ocf_datapipes.batch.fake.fake_batch import fake_data_pipeline
from ocf_datapipes.training.pvnet import pvnet_datapipe

from ocf_datapipes.utils.utils import stack_np_examples_into_batch

from datetime import datetime

def set_fsspec_for_multiprocess() -> None:
    """
    Clear reference to the loop and thread.
    This is a nasty hack that was suggested but NOT recommended by the lead fsspec developer!
    This appears necessary otherwise gcsfs hangs when used after forking multiple worker processes.
    Only required for fsspec >= 0.9.0
    See:
    - https://github.com/fsspec/gcsfs/issues/379#issuecomment-839929801
    - https://github.com/fsspec/filesystem_spec/pull/963#issuecomment-1131709948
    TODO: Try deleting this two lines to make sure this is still relevant.
    """
    fsspec.asyn.iothread[0] = None
    fsspec.asyn.loop[0] = None


def worker_init_fn(worker_id):
    """Configures each dataset worker process.
    1. Get fsspec ready for multi process
    2. To call NowcastingDataset.per_worker_init().
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
        n_train_data,
        n_val_data,
        n_test_data,
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
        self.n_train_data = n_train_data
        self.n_val_data = n_val_data
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
        
    def _get_datapipe(self, start_time, end_time, n_data, batch_size):
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
                .set_length(n_data)
                .map(stack_np_examples_into_batch)
        )
        return data_pipeline
        
    def train_dataloader(self):
        datapipe = self._get_datapipe(*self.train_period, self.n_train_data, self.batch_size)
        return DataLoader(datapipe, **self.dataloader_config)

    def val_dataloader(self):
        datapipe = self._get_datapipe(*self.val_period, self.n_val_data, self.batch_size)
        kwargs = dict(**self.dataloader_config)
        kwargs['num_workers'] = 0
        kwargs['prefetch_factor'] = 2
        kwargs['worker_init_fn'] = None
        kwargs['persistent_workers'] = False
        return DataLoader(datapipe, **kwargs)

    def test_dataloader(self):
        datapipe = self._get_datapipe(*self.test_period, self.n_test_data, self.batch_size)
        kwargs = dict(**self.dataloader_config)
        kwargs['num_workers'] = 0
        kwargs['prefetch_factor'] = 2
        kwargs['worker_init_fn'] = None
        kwargs['persistent_workers'] = False
        return DataLoader(datapipe, **kwargs)

