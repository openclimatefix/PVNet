""" Data module for pytorch lightning """

from ocf_data_sampler.torch_datasets.datasets.pvnet_uk import PVNetUKRegionalDataset
from ocf_data_sampler.torch_datasets.sample.uk_regional import UKRegionalSample
from torch.utils.data import Dataset
from omegaconf import DictConfig
from typing import Optional

from pvnet.data.base_datamodule import BaseDataModule, PremadeSamplesDataset
from pvnet.utils import get_logger


class DataModule(BaseDataModule):
    """Datamodule for training pvnet and using pvnet pipeline in `ocf-data-sampler`."""

    def __init__(
        self,
        configuration: str | None = None,
        sample_dir: str | None = None,
        batch_size: int = 16,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        train_period: list[str | None] = [None, None],
        val_period: list[str | None] = [None, None],
        # THIS ARGUMENT IS IGNORED. Augmentations are now hard-coded.
        train_augmentations_config: Optional[DictConfig] = None, 
    ):
        """Datamodule for training pvnet architecture.

        Can also be used with pre-made batches if `sample_dir` is set.

        Args:
            configuration: Path to configuration file.
            sample_dir: Path to the directory of pre-saved samples. Cannot be used together with
                `configuration` or '[train/val]_period'.
            batch_size: Batch size.
            num_workers: Number of workers to use in multiprocess batch loading.
            prefetch_factor: Number of data will be prefetched at the end of each worker process.
            train_period: Date range filter for train dataloader.
            val_period: Date range filter for val dataloader.
            train_augmentations_config: THIS IS IGNORED.
        """
        super().__init__(
            configuration=configuration,
            sample_dir=sample_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            train_period=train_period,
            val_period=val_period,
        )
        self.log = get_logger(__name__)


    def _get_streamed_samples_dataset(self, start_time, end_time) -> Dataset:
        # This path is for creating samples, so no augmentations are applied.
        return PVNetUKRegionalDataset(
            self.configuration,
            start_time=start_time,
            end_time=end_time,
        )

    def _get_premade_samples_dataset(self, subdir) -> Dataset:
        split_dir = f"{self.sample_dir}/{subdir}"
        
        # Augmentations are now hard-coded on if it's the training set
        apply_augs = (subdir == "train")
        
        if apply_augs:
            self.log.info("HARD-CODED augmentations will be applied to the training set.")
            
        return PremadeSamplesDataset(
            split_dir,
            UKRegionalSample,
            apply_augmentations=apply_augs,
        )
