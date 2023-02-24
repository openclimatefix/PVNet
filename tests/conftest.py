import pytest
from ocf_datapipes.config.model import Configuration

from pvnet.utils import load_config

@pytest.fixture()
def configuration():
    configuration = load_config(
        "tests/testconfigs/datamodule/configuration/test.yaml"
    )
    configuration = Configuration(**configuration)
    
    return configuration
