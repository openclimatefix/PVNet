from predict_pv_yield.models.perceiver.perceiver import PerceiverModel, params
from nowcasting_dataloader.fake import FakeDataset
import torch
from nowcasting_dataset.config.model import Configuration


def test_init_model():
    """Initilize the model"""
    _ = PerceiverModel(history_minutes=3, forecast_minutes=3, nwp_channels=params["nwp_channels"])


def test_model_forward(configuration_perceiver):

    dataset_configuration = configuration_perceiver
    dataset_configuration.input_data.nwp.nwp_image_size_pixels_height = 64
    dataset_configuration.input_data.satellite.satellite_image_size_pixels_height = 16

    model = PerceiverModel(
        history_minutes=30,
        forecast_minutes=60,
        nwp_channels=params["nwp_channels"],
        embedding_dem=2048
    )  # doesnt do anything

    # set up fake data
    train_dataset = FakeDataset(configuration=dataset_configuration)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=None)
    # get data
    x = next(iter(train_dataloader))

    # run data through model
    y = model(x)

    # check out put is the correct shape
    assert len(y.shape) == 2
    assert y.shape[0] == dataset_configuration.process.batch_size
    assert y.shape[1] == 60 // 5
