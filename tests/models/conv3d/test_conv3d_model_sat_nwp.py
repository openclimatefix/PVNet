import pytorch_lightning as pl

from pvnet.models.conv3d.model_sat_nwp import Model
from pvnet.utils import load_config

from ocf_datapipes.config.model import Configuration
from ocf_datapipes.batch.fake.fake_batch import fake_data_pipeline, make_fake_batch
from ocf_datapipes.transform.numpy.batch.add_length import AddLengthIterDataPipe

from torch.utils.data import DataLoader




def test_init():

    config_file = "tests/testconfigs/model/conv3d_sat_nwp.yaml"
    config = load_config(config_file)

    _ = Model(**config)



def test_model_forward(configuration):

    config_file = "tests/testconfigs/model/conv3d_sat_nwp.yaml"
    config = load_config(config_file)
    
    # start model
    model = Model(**config)

    data_config: Configuration = configuration

    # run data through model
    batch = make_fake_batch(configuration=data_config, to_torch=True)
    y = model(batch)

    # check out put is the correct shape
    assert len(y.shape) == 2
    assert y.shape[0] == 2
    assert y.shape[1] == model.forecast_len_30


def test_model_forward_no_satellite(configuration):

    config_file = "tests/testconfigs/model/conv3d_sat_nwp.yaml"
    config = load_config(config_file)
    config["include_future_satellite"] = False

    # start model
    model = Model(**config)

    data_config: Configuration = configuration

    # run data through model
    data_pipeline = fake_data_pipeline(configuration=data_config)
    train_dataloader = DataLoader(data_pipeline, batch_size=None)
    batch = next(iter(train_dataloader))

    y = model(batch)

    # check out put is the correct shape
    assert len(y.shape) == 2
    assert y.shape[0] == 2
    assert y.shape[1] == model.forecast_len_30


def test_train(configuration):

    config_file = "tests/testconfigs/model/conv3d_sat_nwp.yaml"
    config = load_config(config_file)

    # start model
    model = Model(**config)

    data_config: Configuration = configuration

    # create fake data loader
    data_pipeline = AddLengthIterDataPipe(source_datapipe=fake_data_pipeline(configuration=data_config), length=2)
    train_dataloader = DataLoader(data_pipeline, batch_size=None)

    # fit model
    trainer = pl.Trainer(max_epochs=1, max_steps=2)
    trainer.fit(model, train_dataloader)

    # predict over training set
    _ = trainer.predict(model, train_dataloader)
