from predict_pv_yield.models.conv3d.model_sat_nwp import Model
import torch
import pytorch_lightning as pl
from predict_pv_yield.utils import load_config
from nowcasting_dataloader.fake import FakeDataset
from nowcasting_dataset.config.model import Configuration


def test_init():

    config_file = "tests/configs/model/conv3d_sat_nwp.yaml"
    config = load_config(config_file)

    _ = Model(**config)


def test_model_forward(configuration_conv3d):

    config_file = "tests/configs/model/conv3d_sat_nwp.yaml"
    config = load_config(config_file)

    # start model
    model = Model(**config)

    dataset_configuration: Configuration = configuration_conv3d
    dataset_configuration.input_data.nwp.nwp_image_size_pixels_height = 16
    dataset_configuration.input_data.nwp.nwp_image_size_pixels_width = 16

    dataset_configuration.input_data.satellite.satellite_image_size_pixels_height = 16
    dataset_configuration.input_data.satellite.satellite_image_size_pixels_width = 16

    # create fake data loader
    train_dataset = FakeDataset(configuration=dataset_configuration)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=None)
    x = next(iter(train_dataloader))

    # run data through model
    y = model(x)

    # check out put is the correct shape
    assert len(y.shape) == 2
    assert y.shape[0] == 2
    assert y.shape[1] == model.forecast_len_30


def test_model_forward_no_satellite(configuration_conv3d):

    config_file = "tests/configs/model/conv3d_sat_nwp.yaml"
    config = load_config(config_file)
    config["include_future_satellite"] = False

    # start model
    model = Model(**config)

    dataset_configuration = configuration_conv3d
    dataset_configuration.input_data.nwp.nwp_image_size_pixels_height = 16

    # create fake data loader
    train_dataset = FakeDataset(configuration=dataset_configuration)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=None)
    x = next(iter(train_dataloader))

    # run data through model
    y = model(x)

    # check out put is the correct shape
    assert len(y.shape) == 2
    assert y.shape[0] == 2
    assert y.shape[1] == model.forecast_len_30


def test_train(configuration_conv3d):

    config_file = "tests/configs/model/conv3d_sat_nwp.yaml"
    config = load_config(config_file)

    dataset_configuration = configuration_conv3d
    dataset_configuration.input_data.nwp.nwp_image_size_pixels_height = 16

    # start model
    model = Model(**config)

    # create fake data loader
    train_dataset = FakeDataset(configuration=dataset_configuration)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=None)

    # fit model
    trainer = pl.Trainer(gpus=0, max_epochs=1)
    trainer.fit(model, train_dataloader)

    # predict over training set
    _ = trainer.predict(model, train_dataloader)
