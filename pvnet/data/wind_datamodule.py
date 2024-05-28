""" Data module for pytorch lightning """
import glob

from ocf_datapipes.batch import BatchKey, batch_to_tensor, stack_np_examples_into_batch
from ocf_datapipes.training.windnet import windnet_netcdf_datapipe

from pvnet.data.base import BaseDataModule


class WindDataModule(BaseDataModule):
    """Datamodule for training windnet and using windnet pipeline in `ocf_datapipes`."""

    def _get_datapipe(self, start_time, end_time):
        # TODO is this is not right, need to load full windnet pipeline
        data_pipeline = windnet_netcdf_datapipe(
            self.configuration,
            keys=["wind", "nwp", "sensor"],
        )

        data_pipeline = (
            data_pipeline.batch(self.batch_size)
            .map(stack_np_examples_into_batch)
            .map(batch_to_tensor)
        )
        return data_pipeline

    def _get_premade_batches_datapipe(self, subdir, shuffle=False):
        filenames = list(glob.glob(f"{self.batch_dir}/{subdir}/*.nc"))
        data_pipeline = windnet_netcdf_datapipe(
            keys=["wind", "nwp", "sensor"],
            filenames=filenames,
            nwp_channels=self.nwp_channels,
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
                .split_batches(splitting_key=BatchKey.wind)
                .shuffle(buffer_size=self.shuffle_factor * self.batch_size)
            )
        else:
            data_pipeline = (
                data_pipeline.sharding_filter()
                # Split the batches so we can use any batch-size
                .split_batches(splitting_key=BatchKey.wind)
            )

        data_pipeline = (
            data_pipeline.batch(self.batch_size)
            .map(stack_np_examples_into_batch)
            .map(batch_to_tensor)
            .set_length(int(len(filenames) / self.batch_size))
        )

        return data_pipeline
