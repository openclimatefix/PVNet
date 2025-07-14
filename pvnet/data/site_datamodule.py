""" Data module for pytorch lightning """

from ocf_data_sampler.torch_datasets.datasets.site import SitesDataset
from ocf_data_sampler.torch_datasets.sample.site import SiteSample
from torch.utils.data import Dataset

from pvnet.data.base_datamodule import (
    BasePresavedDataModule,
    BaseStreamedDataModule,
    PresavedSamplesDataset,
)


class SitePresavedDataModule(BasePresavedDataModule):
    """Datamodule for loading pre-saved samples."""

    def _get_premade_samples_dataset(self, subdir: str) -> Dataset:
        split_dir = f"{self.sample_dir}/{subdir}"
        return PresavedSamplesDataset(split_dir, SiteSample)


class SiteStreamedDataModule(BaseStreamedDataModule):
    """Datamodule which streams samples using sampler for ocf-data-sampler."""

    def _get_streamed_samples_dataset(
        self,
        start_time: str | None,
        end_time: str | None
    ) -> Dataset:
        return SitesDataset(self.configuration, start_time=start_time, end_time=end_time)
