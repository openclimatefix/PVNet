""" Data module for pytorch lightning """

import resource

import torch
from ocf_datapipes.batch import batch_to_tensor, stack_np_examples_into_batch
from ocf_datapipes.training.pvnet import pvnet_datapipe
from torch.utils.data.datapipes.iter import FileLister

from pvnet.data.base import BaseDataModule

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


class DataModule(BaseDataModule):
    """Datamodule for training pvnet and using pvnet pipeline in `ocf_datapipes`."""

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
                .shuffle(buffer_size=self.shuffle_factor * self.batch_size)
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
