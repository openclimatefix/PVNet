""" Data module for pytorch lightning """

from ocf_data_sampler.torch_datasets.datasets.pvnet_uk import PVNetUKRegionalDataset
from ocf_data_sampler.torch_datasets.sample.uk_regional import UKRegionalSample
from torch.utils.data import Dataset

from pvnet.data.base_datamodule import (
    BasePresavedDataModule,
    BaseStreamedDataModule,
    PresavedSamplesDataset,
)


class UKRegionalPresavedDataModule(BasePresavedDataModule):
    """Datamodule for loading pre-saved samples."""

    def _get_premade_samples_dataset(self, subdir: str) -> Dataset:
        split_dir = f"{self.sample_dir}/{subdir}"
        return PresavedSamplesDataset(split_dir, UKRegionalSample)


class UKRegionalStreamedDataModule(BaseStreamedDataModule):
    """Datamodule which streams samples using sampler for ocf-data-sampler."""

    def _get_streamed_samples_dataset(
        self,
        start_time: str | None,
        end_time: str | None
    ) -> Dataset:
        return PVNetUKRegionalDataset(self.configuration, start_time=start_time, end_time=end_time)
