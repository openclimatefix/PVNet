from pvnet.data.datamodule import DataModule
from pvnet.data.wind_datamodule import WindDataModule
from pvnet.data.pv_site_datamodule import PVSiteDataModule
import os
from ocf_datapipes.batch.batches import BatchKey, NWPBatchKey


def test_init():
    dm = DataModule(
        configuration=None,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
        train_period=[None, None],
        val_period=[None, None],
        test_period=[None, None],
        batch_dir="tests/test_data/sample_batches",
    )


def test_wind_init():
    dm = WindDataModule(
        configuration=None,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
        train_period=[None, None],
        val_period=[None, None],
        test_period=[None, None],
        batch_dir="tests/data/sample_batches",
    )


def test_wind_init_with_nwp_filter():
    dm = WindDataModule(
        configuration=None,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
        train_period=[None, None],
        val_period=[None, None],
        test_period=[None, None],
        batch_dir="tests/test_data/sample_wind_batches",
        nwp_channels={"ecmwf": ["t2m", "v200"]},
    )
    dataloader = iter(dm.train_dataloader())

    batch = next(dataloader)
    batch_channels = batch[BatchKey.nwp]["ecmwf"][NWPBatchKey.nwp_channel_names]
    print(batch_channels)
    for v in ["t2m", "v200"]:
        assert v in batch_channels
    assert batch[BatchKey.nwp]["ecmwf"][NWPBatchKey.nwp].shape[2] == 2


def test_pv_site_init():
    dm = PVSiteDataModule(
        configuration=f"{os.path.dirname(os.path.abspath(__file__))}/test_data/sample_batches/data_configuration.yaml",
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
        train_period=[None, None],
        val_period=[None, None],
        test_period=[None, None],
        batch_dir=None,
    )


def test_iter():
    dm = DataModule(
        configuration=None,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
        train_period=[None, None],
        val_period=[None, None],
        test_period=[None, None],
        batch_dir="tests/test_data/sample_batches",
    )

    batch = next(iter(dm.train_dataloader()))


def test_iter_multiprocessing():
    dm = DataModule(
        configuration=None,
        batch_size=2,
        num_workers=2,
        prefetch_factor=2,
        train_period=[None, None],
        val_period=[None, None],
        test_period=[None, None],
        batch_dir="tests/test_data/sample_batches",
    )

    batch = next(iter(dm.train_dataloader()))
    for batch in dm.train_dataloader():
        pass
