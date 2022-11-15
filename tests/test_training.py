import os

import tilemapbase
from hydra import compose, initialize

from pvnet.training import train


def test_train():

    os.environ["NEPTUNE_API_TOKEN"] = "not_a_token"

    # for Github actions need to create this
    tilemapbase.init(create=True)

    initialize(config_path="../configs", job_name="test_app")
    config = compose(
        config_name="config",
        overrides=[
            "logger=csv",
            "experiment=example_simple",
            "datamodule.fake_data=true",
            "datamodule.data_path=tests/configs/dataset",
            "trainer.fast_dev_run=true",
        ],
    )

    train(config=config)
