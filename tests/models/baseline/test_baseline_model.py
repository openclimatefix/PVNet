from pvnet.models.baseline.last_value import Model
import pytorch_lightning as pl
from ocf_datapipes.batch.fake.fake_batch import fake_data_pipeline, make_fake_batch
from ocf_datapipes.transform.numpy.batch.add_length import AddLengthIterDataPipe

from torch.utils.data import DataLoader


def test_init():
    _ = Model()


def test_model_forward(configuration):
    # start model
    model = Model(forecast_minutes=configuration.input_data.default_forecast_minutes)

    # satellite data
    x = make_fake_batch(configuration=configuration, to_torch=True)

    # run data through model
    y = model(x)

    # check out put is the correct shape
    assert len(y.shape) == 2
    assert y.shape[0] == configuration.process.batch_size
    assert y.shape[1] == configuration.input_data.default_forecast_minutes // 5


def test_trainer(configuration):
    # start model
    model = Model(forecast_minutes=configuration.input_data.default_forecast_minutes)

    # create fake data loader
    data_pipeline = AddLengthIterDataPipe(
        source_datapipe=fake_data_pipeline(configuration=configuration), length=2
    )
    train_dataloader = DataLoader(data_pipeline, batch_size=None)

    # set up trainer
    trainer = pl.Trainer(gpus=0, max_epochs=1)

    # test over training set
    _ = trainer.test(model, train_dataloader)
