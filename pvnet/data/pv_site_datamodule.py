""" Data module for pytorch lightning """
import glob

from ocf_datapipes.batch import BatchKey, batch_to_tensor, stack_np_examples_into_batch
from ocf_datapipes.training.pvnet_site import (
    pvnet_site_datapipe,
    pvnet_site_netcdf_datapipe,
    split_dataset_dict_dp,
    uncombine_from_single_dataset,
)

from pvnet.data.base import BaseDataModule


class PVSiteDataModule(BaseDataModule):
    """Datamodule for training pvnet site and using pvnet site pipeline in `ocf_datapipes`."""

    def _get_datapipe(self, start_time, end_time):
        data_pipeline = pvnet_site_datapipe(
            self.configuration,
            start_time=start_time,
            end_time=end_time,
        )
        data_pipeline = data_pipeline.map(uncombine_from_single_dataset).map(split_dataset_dict_dp)
        data_pipeline = data_pipeline.pvnet_site_convert_to_numpy_batch()

        data_pipeline = (
            data_pipeline.batch(self.batch_size)
            .map(stack_np_examples_into_batch)
            .map(batch_to_tensor)
        )
        return data_pipeline

    def _get_premade_batches_datapipe(self, subdir, shuffle=False):
        filenames = list(glob.glob(f"{self.batch_dir}/{subdir}/*.nc"))
        data_pipeline = pvnet_site_netcdf_datapipe(
            keys=["pv", "nwp"],  # add other keys e.g. sat if used as input in site model
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
