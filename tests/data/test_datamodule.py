from pvnet.data import DataModule, SiteDataModule
import os
import pytest


def test_init():
    dm = DataModule(
        configuration=None,
        sample_dir="tests/test_data/presaved_samples_uk_regional",
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
        train_period=[None, None],
        val_period=[None, None],
    )


def test_iter():
    dm = DataModule(
        configuration=None,
        sample_dir="tests/test_data/presaved_samples_uk_regional",
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
        train_period=[None, None],
        val_period=[None, None],
    )


def test_iter_multiprocessing():
    dm = DataModule(
        configuration=None,
        sample_dir="tests/test_data/presaved_samples_uk_regional",
        batch_size=1,
        num_workers=2,
        prefetch_factor=1,
        train_period=[None, None],
        val_period=[None, None],
    )

    served_batches = 0
    for batch in dm.train_dataloader():
        served_batches += 1

        # Stop once we've got 2 batches
        if served_batches == 2:
            break

    # Make sure we've served 2 batches
    assert served_batches == 2


def test_site_init_sample_dir():
    dm = SiteDataModule(
        configuration=None,
        sample_dir="tests/test_data/presaved_site_samples",
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
        train_period=[None, None],
        val_period=[None, None],
    )


def test_site_init_config():
    dm = SiteDataModule(
        configuration=f"{os.path.dirname(os.path.abspath(__file__))}/test_data/presaved_site_samples/data_configuration.yaml",
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
        train_period=[None, None],
        val_period=[None, None],
        sample_dir=None,
    )

