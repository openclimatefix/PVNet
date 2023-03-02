import os

import tilemapbase
from hydra import compose, initialize

from pvnet.training import train


def test_train():

    os.environ["NEPTUNE_API_TOKEN"] = "not_a_token"

    # for Github actions need to create this
    tilemapbase.init(create=True)

    initialize(config_path="testconfigs/", job_name="test_app")
    config = compose(
        config_name="config.yaml",
        overrides=[
            "model=conv3d_sat_nwp.yaml",
        ],
    )

    train(config=config)