import pytorch_lightning as pl
import torch
from nowcasting_dataloader.fake import FakeDataset

from pvnet.models.conv3d.model_sat_nwp import Model
from pvnet.utils import load_config

from ocf_datapipes.config.model import Configuration
from ocf_datapipes.batch.fake.fake_batch import fake_data_pipeline

from torch.utils.data import DataLoader


# def test_init():
#
#     config_file = "tests/configs/model/conv3d_sat_nwp.yaml"
#     config = load_config(config_file)
#
#     _ = Model(**config)
#
#
# def test_model_forward(configuration_conv3d):
#
#     config_file = "tests/configs/model/conv3d_sat_nwp.yaml"
#     config = load_config(config_file)
#
#     # start model
#     model = Model(**config)
#
#     dataset_configuration: Configuration = configuration_conv3d
#     dataset_configuration.input_data.nwp.nwp_image_size_pixels_height = 16
#     dataset_configuration.input_data.nwp.nwp_image_size_pixels_width = 16
#
#     dataset_configuration.input_data.satellite.satellite_image_size_pixels_height = 16
#     dataset_configuration.input_data.satellite.satellite_image_size_pixels_width = 16
#
#     # run data through model
#     batch = make_fake_batch(configuration=dataset_configuration, to_torch=True)
#     y = model(batch)
#
#     # check out put is the correct shape
#     assert len(y.shape) == 2
#     assert y.shape[0] == 2
#     assert y.shape[1] == model.forecast_len_30


def test_model_forward_no_satellite(configuration_conv3d):

    config_file = "tests/configs/model/conv3d_sat_nwp.yaml"
    config = load_config(config_file)
    config["include_future_satellite"] = False

    # start model
    model = Model(**config)

    dataset_configuration: Configuration = configuration_conv3d
    dataset_configuration.input_data.nwp.nwp_image_size_pixels_height = 16
    dataset_configuration.input_data.nwp.nwp_image_size_pixels_width = 16
    dataset_configuration.input_data.nwp.time_resolution_minutes = 60
    dataset_configuration.input_data.nwp.forecast_minutes = 60
    dataset_configuration.input_data.nwp.history_minutes = 60
    dataset_configuration.input_data.nwp.nwp_channels = dataset_configuration.input_data.nwp.nwp_channels[0:10]
    dataset_configuration.input_data.satellite.satellite_image_size_pixels_height = 16
    dataset_configuration.input_data.satellite.satellite_image_size_pixels_width = 16
    dataset_configuration.input_data.pv.n_pv_systems_per_example = 128

    # run data through model
    data_pipeline = fake_data_pipeline(configuration=dataset_configuration)
    train_dataloader = DataLoader(data_pipeline, batch_size=None)
    batch = next(iter(train_dataloader))

    y = model(batch)

    # check out put is the correct shape
    assert len(y.shape) == 2
    assert y.shape[0] == 2
    assert y.shape[1] == model.forecast_len_30

#
# def test_train(configuration_conv3d):
#
#     config_file = "tests/configs/model/conv3d_sat_nwp.yaml"
#     config = load_config(config_file)
#
#     dataset_configuration: Configuration = configuration_conv3d
#     dataset_configuration.input_data.nwp.nwp_image_size_pixels_height = 16
#     dataset_configuration.input_data.nwp.nwp_image_size_pixels_width = 16
#
#     dataset_configuration.input_data.satellite.satellite_image_size_pixels_height = 16
#     dataset_configuration.input_data.satellite.satellite_image_size_pixels_width = 16
#
#     # start model
#     model = Model(**config)
#
#     # create fake data loader
#     train_dataset = FakeDataset(configuration=dataset_configuration)
#     train_dataloader = DataLoader(train_dataset, batch_size=None)
#
#     # fit model
#     trainer = pl.Trainer(max_epochs=1)
#     trainer.fit(model, train_dataloader)
#
#     # predict over training set
#     _ = trainer.predict(model, train_dataloader)
