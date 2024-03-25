""" Data module for pytorch lightning """

import glob

from ocf_datapipes.batch import BatchKey, stack_np_examples_into_batch
from ocf_datapipes.training.pvnet_site import pvnet_site_netcdf_datapipe

from pvnet.data.base import BaseDataModule
from pvnet.data.utils import batch_to_tensor


class PVSiteDataModule(BaseDataModule):
    """Datamodule for training pvnet site and using pvnet site pipeline in `ocf_datapipes`."""

    def _get_datapipe(self, start_time, end_time):
        data_pipeline = pvnet_site_netcdf_datapipe(
            keys=["pv", "nwp"],
        )

        data_pipeline = (
            data_pipeline.batch(self.batch_size)
            .map(stack_np_examples_into_batch)
            .map(batch_to_tensor)
        )
        return data_pipeline

    def _get_premade_batches_datapipe(self, subdir, shuffle=False):
        filenames = list(glob.glob(f"{self.batch_dir}/{subdir}/*.nc"))
        data_pipeline = pvnet_site_netcdf_datapipe(
            keys=["pv", "nwp"],
            filenames=filenames,
        )
        data_pipeline = (
            data_pipeline.batch(self.batch_size)
            .map(stack_np_examples_into_batch)
            .map(batch_to_tensor)
        )
        if shuffle:
            data_pipeline = (
                data_pipeline.shuffle(buffer_size=100)
                .sharding_filter()
                # Split the batches and reshuffle them to be combined into new batches
                .split_batches(splitting_key=BatchKey.pv)
                .shuffle(buffer_size=self.shuffle_factor * self.batch_size)
            )
        else:
            data_pipeline = (
                data_pipeline.sharding_filter()
                # Split the batches so we can use any batch-size
                .split_batches(splitting_key=BatchKey.pv)
            )

        data_pipeline = (
            data_pipeline.batch(self.batch_size)
            .map(stack_np_examples_into_batch)
            .map(batch_to_tensor)
            .set_length(int(len(filenames) / self.batch_size))
        )

        return data_pipeline
