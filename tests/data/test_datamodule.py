from pvnet.data.datamodule import DataModule


def test_init():
    dm = DataModule(
        configuration=None,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
        train_period=[None, None],
        val_period=[None, None],
        test_period=[None, None],
        block_nwp_and_sat=False,
        batch_dir="tests/test_data/sample_batches",
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
        block_nwp_and_sat=False,
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
        block_nwp_and_sat=False,
        batch_dir="tests/test_data/sample_batches",
    )

    batch = next(iter(dm.train_dataloader()))
    for batch in dm.train_dataloader():
        pass

