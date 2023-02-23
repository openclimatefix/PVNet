import pytest
from ocf_datapipes.config.model import Configuration

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

    config_file = "tests/configs/model/conv3d_sat_nwp.yaml"
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
    
    # aliases for readability below
    nwp = dataset_configuration.input_data.nwp
    sat = dataset_configuration.input_data.satellite
    pv = dataset_configuration.input_data.pv
    gsp = dataset_configuration.input_data.gsp
    
    nwp.nwp_image_size_pixels_height = config['nwp_image_size_pixels_height'] # 16
    nwp.nwp_image_size_pixels_width = config['nwp_image_size_pixels_height'] # 16
    nwp.time_resolution_minutes = 60
    nwp.history_minutes = config['history_minutes'] # 60
    nwp.forecast_minutes = config['forecast_minutes'] # 60
    nwp.nwp_channels = nwp.nwp_channels[0:config["number_nwp_channels"]]
    
    sat.satellite_image_size_pixels_height = config['image_size_pixels'] # 16
    sat.satellite_image_size_pixels_width = config['image_size_pixels'] # 16
    sat.history_minutes = config['history_minutes'] # 60
    sat.forecast_minutes = config['forecast_minutes'] # 60
    
    pv.n_pv_systems_per_example = 128
    pv.history_minutes = config['history_minutes'] # 60
    pv.forecast_minutes = config['forecast_minutes'] # 60
    
    gsp.history_minutes = config['history_minutes'] # 60
    gsp.forecast_minutes = config['forecast_minutes'] # 60
    gsp.time_resolution_minutes = 30

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
