""" Data module for pytorch lightning """
from glob import glob

from pvnet.data.base_datamodule import BaseDataModule
from ocf_data_sampler.torch_datasets.site import SitesDataset, convert_netcdf_to_numpy_sample

from torch.utils.data import Dataset
import xarray as xr


class NetcdfPreMadeSamplesDataset(Dataset):
    """Dataset to load pre-made netcdf samples"""

    def __init__(
        self,
        sample_dir,
    ):
        """Dataset to load pre-made netcdf samples

        Args:
            sample_dir: Path to the directory of pre-saved samples.
        """
        self.sample_paths = glob(f"{sample_dir}/*.nc")

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        # open the sample
        ds = xr.open_dataset(self.sample_paths[idx])

        # convert to numpy
        sample = convert_netcdf_to_numpy_sample(ds)
        return sample

class SiteDataModule(BaseDataModule):
    """Datamodule for training pvnet and using pvnet pipeline in `ocf_datapipes`."""

    def __init__(
        self,
        configuration: str | None = None,
        sample_dir: str | None = None,
        batch_size: int = 16,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        train_period: list[str | None] = [None, None],
        val_period: list[str | None] = [None, None],
    ):
        """Datamodule for training pvnet architecture.

        Can also be used with pre-made batches if `sample_dir` is set.

        Args:
            configuration: Path to datapipe configuration file.
            sample_dir: Path to the directory of pre-saved samples. Cannot be used together with
                `configuration` or '[train/val]_period'.
            batch_size: Batch size.
            num_workers: Number of workers to use in multiprocess batch loading.
            prefetch_factor: Number of data will be prefetched at the end of each worker process.
            train_period: Date range filter for train dataloader.
            val_period: Date range filter for val dataloader.

        """
        super().__init__(
            configuration=configuration,
            sample_dir=sample_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            train_period=train_period,
            val_period=val_period
        )

    def _get_streamed_samples_dataset(self, start_time, end_time) -> Dataset:
        return SitesDataset(self.configuration, start_time=start_time, end_time=end_time)

    def _get_premade_samples_dataset(self, subdir) -> Dataset:
       split_dir = f"{self.sample_dir}/{subdir}"
       return NetcdfPreMadeSamplesDataset(split_dir)
