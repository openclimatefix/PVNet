from pvnet.data.datamodule import DataModule


def test_init():
    dm = DataModule(
        configuration=None,
        sample_dir="tests/test_data/presaved_samples",
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
        train_period=[None, None],
        val_period=[None, None],
    )


def test_iter():
    dm = DataModule(
        configuration=None,
        sample_dir="tests/test_data/presaved_samples",
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
        train_period=[None, None],
        val_period=[None, None],
    )

    batch = next(iter(dm.train_dataloader()))


def test_iter_multiprocessing():
    dm = DataModule(
        configuration=None,
        sample_dir="tests/test_data/presaved_samples",
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


# TODO add test cases with some netcdfs premade samples
