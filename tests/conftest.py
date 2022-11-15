import pytest
from nowcasting_dataset.config.model import Configuration

from pvnet.utils import load_config


@pytest.fixture()
def configuration():
    configuration = Configuration()
    configuration.input_data = configuration.input_data.set_all_to_defaults()
    configuration.process.batch_size = 2
    configuration.input_data.default_history_minutes = 30
    configuration.input_data.default_forecast_minutes = 60
    configuration.input_data.nwp.nwp_image_size_pixels_height = 16

    return configuration


@pytest.fixture()
def configuration_conv3d():

    config_file = "tests/configs/model/conv3d.yaml"
    config = load_config(config_file)

    dataset_configuration = Configuration()
    dataset_configuration.process.batch_size = 2
    dataset_configuration.input_data.default_history_minutes = config["history_minutes"]
    dataset_configuration.input_data.default_forecast_minutes = config[
        "forecast_minutes"
    ]
    dataset_configuration.input_data = (
        dataset_configuration.input_data.set_all_to_defaults()
    )
    dataset_configuration.input_data.nwp.nwp_image_size_pixels_height = 2
    dataset_configuration.input_data.satellite.satellite_image_size_pixels_height = (
        config["image_size_pixels"]
    )
    dataset_configuration.input_data.satellite.forecast_minutes = config[
        "forecast_minutes"
    ]
    dataset_configuration.input_data.satellite.history_minutes = config[
        "history_minutes"
    ]

    return dataset_configuration


@pytest.fixture()
def configuration_perceiver():

    dataset_configuration = Configuration()
    dataset_configuration.input_data = (
        dataset_configuration.input_data.set_all_to_defaults()
    )
    dataset_configuration.process.batch_size = 2
    dataset_configuration.input_data.nwp.nwp_image_size_pixels_height = 16
    dataset_configuration.input_data.satellite.satellite_image_size_pixels_height = 16
    dataset_configuration.input_data.default_history_minutes = 30
    dataset_configuration.input_data.default_forecast_minutes = 120
    dataset_configuration.input_data.nwp.nwp_channels = (
        dataset_configuration.input_data.nwp.nwp_channels[0:10]
    )

    return dataset_configuration
