import os

import hydra
from hydra import compose, initialize

from pvnet.utils import extras, print_config


def test_utils():
    """
    Test that util functions work. This just runs them. Perhaps slightly harder to check they work how they should.
    """
    os.environ["NEPTUNE_API_TOKEN"] = "not_a_token"

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path="../configs", job_name="test_app")
    config = compose(config_name="config")

    extras(config)

    print_config(config)
