from predict_pv_yield.models.perceiver.perceiver_nwp_sat import Model, params, TOTAL_SEQ_LEN
from predict_pv_yield.data.dataloader import FakeDataset
import torch


def test_init_model():
    """Initilize the model"""
    _ = Model(
        history_minutes=3, forecast_minutes=3, nwp_channels=params["nwp_channels"], output_variable="gsp_yield"
    )


def test_model_forward(configuration_perceiver):

    dataset_configuration = configuration_perceiver
    dataset_configuration.process.batch_size = 2
    dataset_configuration.input_data.nwp.nwp_image_size_pixels_height = 16
    dataset_configuration.input_data.satellite.satellite_image_size_pixels_height = 16

    model = Model(
        history_minutes=30,
        forecast_minutes=60,
        nwp_channels=params["nwp_channels"],
        output_variable="gsp_yield",
    )  # doesnt do anything

    batch_size = 2
    # set up fake data
    train_dataset = FakeDataset(configuration=dataset_configuration)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=None)
    # get data
    x = next(iter(train_dataloader))

    # run data through model
    y = model(x)

    # check out put is the correct shape
    assert len(y.shape) == 2
    assert y.shape[0] == batch_size
    assert y.shape[1] == 60 // 30
