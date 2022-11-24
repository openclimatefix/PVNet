from pvnet.models.baseline.last_value import Model
import pytorch_lightning as pl
import pandas as pd

import tempfile

from ocf_datapipes.batch.fake.fake_batch import fake_data_pipeline, make_fake_batch
from ocf_datapipes.transform.numpy.batch.add_length import AddLengthIterDataPipe

from torch.utils.data import DataLoader



def test_init():

    _ = Model(output_variable="gsp_yield")


def test_model_forward(configuration):

    # start model
    model = Model(
        forecast_minutes=configuration.input_data.default_forecast_minutes,
        history_minutes=configuration.input_data.default_history_minutes,
        output_variable="gsp_yield",
    )

    # satellite data
    x = make_fake_batch(configuration=configuration, to_torch=True)

    # run data through model
    y = model(x)

    # check out put is the correct shape
    assert len(y.shape) == 2
    assert y.shape[0] == configuration.process.batch_size
    assert y.shape[1] == configuration.input_data.default_forecast_minutes // 30


def test_model_validation(configuration):

    # start model
    model = Model(
        forecast_minutes=configuration.input_data.default_forecast_minutes,
        history_minutes=configuration.input_data.default_history_minutes,
        output_variable="gsp_yield",
    )

    # satellite data
    x = make_fake_batch(configuration=configuration, to_torch=True)

    # run data through model
    model.validation_step(x, 0)


def test_trainer(configuration):

    # start model
    model = Model(
        forecast_minutes=configuration.input_data.default_forecast_minutes,
        history_minutes=configuration.input_data.default_history_minutes,
        output_variable="gsp_yield",
    )

    # create fake data loader
    data_pipeline = AddLengthIterDataPipe(source_datapipe=fake_data_pipeline(configuration=configuration), length=2)
    train_dataloader = DataLoader(data_pipeline, batch_size=None)

    # set up trainer
    trainer = pl.Trainer(gpus=0, max_epochs=1)

    # test over training set
    _ = trainer.test(model, train_dataloader)


def test_trainer_validation(configuration):

    # start model
    model = Model(
        forecast_minutes=configuration.input_data.default_forecast_minutes,
        history_minutes=configuration.input_data.default_history_minutes,
        output_variable="gsp_yield",
    )

    # create fake data loader
    data_pipeline = AddLengthIterDataPipe(source_datapipe=fake_data_pipeline(configuration=configuration), length=2)
    train_dataloader = DataLoader(data_pipeline, batch_size=None)

    # set up trainer
    trainer = pl.Trainer(gpus=0, max_epochs=1)

    with tempfile.TemporaryDirectory() as tmpdirname:
        model.results_file_name = f'{tmpdirname}/temp'

        # test over validation set
        _ = trainer.validate(model, train_dataloader)

        # check csv file of validation results has been made
        results_df = pd.read_csv(f'{model.results_file_name}_0.csv')

        assert len(results_df) == len(train_dataloader) * configuration.process.batch_size * model.forecast_len_30
        assert 't0_datetime_utc' in results_df.keys()
        assert 'target_datetime_utc' in results_df.keys()
        assert 'gsp_id' in results_df.keys()
        assert "actual_gsp_pv_outturn_mw" in results_df.keys()
        assert "forecast_gsp_pv_outturn_mw" in results_df.keys()
