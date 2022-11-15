import logging
import os

import torch
from nowcasting_dataloader.datasets import NetCDFDataset, worker_init_fn
from nowcasting_dataloader.fake import FakeDataset
from nowcasting_dataset.config.load import load_yaml_configuration
from pytorch_lightning import LightningDataModule

_LOG = logging.getLogger(__name__)
_LOG.setLevel(logging.DEBUG)

torch.set_default_dtype(torch.float32)


def get_dataloaders(
    n_train_data: int = 24900,
    n_validation_data: int = 900,
    cloud: str = "gcp",
    temp_path=".",
    data_path="prepared_ML_training_data/v4/",
) -> tuple:

    # configuration = load_yaml_configuration(filename=f"{data_path}/configuration.yaml")

    data_module = NetCDFDataModule(
        temp_path=temp_path,
        data_path=data_path,
        cloud=cloud,
        n_train_data=n_train_data,
        n_val_data=n_validation_data,
    )

    train_dataloader = data_module.train_dataloader()
    validation_dataloader = data_module.val_dataloader()

    return train_dataloader, validation_dataloader


class NetCDFDataModule(LightningDataModule):
    """
    Example of LightningDataModule for NETCDF dataset.
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        temp_path: str = ".",
        n_train_data: int = 24900,
        n_val_data: int = 1000,
        cloud: str = "aws",
        num_workers: int = 8,
        pin_memory: bool = True,
        data_path="prepared_ML_training_data/v4/",
        fake_data: bool = False,
    ):
        """
        fake_data: random data is created and used instead. This is useful for testing
        """
        super().__init__()

        self.temp_path = temp_path
        self.data_path = data_path
        self.cloud = cloud
        self.n_train_data = n_train_data
        self.n_val_data = n_val_data
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.fake_data = fake_data

        filename = os.path.join(data_path, "configuration.yaml")
        _LOG.debug(f"Will be loading the configuration file {filename}")
        self.configuration = load_yaml_configuration(filename=filename)

        self.dataloader_config = dict(
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            prefetch_factor=8,
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
            # Disable automatic batching because dataset
            # returns complete batches.
            batch_size=None,
        )

    def train_dataloader(self):
        if self.fake_data:
            train_dataset = FakeDataset(configuration=self.configuration)
        else:
            train_dataset = NetCDFDataset(
                self.n_train_data,
                os.path.join(self.data_path, "train"),
                os.path.join(self.temp_path, "train"),
                configuration=self.configuration,
            )

        return torch.utils.data.DataLoader(train_dataset, **self.dataloader_config)

    def val_dataloader(self):
        if self.fake_data:
            val_dataset = FakeDataset(configuration=self.configuration)
        else:
            val_dataset = NetCDFDataset(
                self.n_val_data,
                os.path.join(self.data_path, "test"),
                os.path.join(self.temp_path, "test"),
                configuration=self.configuration,
            )

        return torch.utils.data.DataLoader(val_dataset, **self.dataloader_config)

    def test_dataloader(self):
        if self.fake_data:
            test_dataset = FakeDataset(configuration=self.configuration)
        else:
            # TODO need to change this to a test folder
            test_dataset = NetCDFDataset(
                self.n_val_data,
                os.path.join(self.data_path, "test"),
                os.path.join(self.temp_path, "test"),
                configuration=self.configuration,
            )

        return torch.utils.data.DataLoader(test_dataset, **self.dataloader_config)
