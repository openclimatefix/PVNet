import os

import pytest
import pandas as pd
import numpy as np
import xarray as xr
import torch
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.models.base import Base_Forecast, Base_PV

from ocf_datapipes.utils.consts import BatchKey
from testcontainers.postgres import PostgresContainer
from datetime import timedelta

import pvnet
from pvnet.data.datamodule import DataModule
from nowcasting_datamodel.models import (
    ForecastSQL,
    GSPYield,
    Location,
    LocationSQL,
)
from nowcasting_datamodel.fake import make_fake_me_latest

xr.set_options(keep_attrs=True)


def time_before_present(dt: timedelta):
    return pd.Timestamp.now(tz=None) - dt


@pytest.fixture(scope="session")
def engine_url():
    """Database engine, this includes the table creation."""
    with PostgresContainer("postgres:14.5") as postgres:
        url = postgres.get_connection_url()
        os.environ["DB_URL"] = url

        database_connection = DatabaseConnection(url, echo=True)

        engine = database_connection.engine

        # Would like to do this here but found the data
        # was not being deleted when using 'db_connection'
        # database_connection.create_all()
        # Base_PV.metadata.create_all(engine)

        yield url

        # Base_PV.metadata.drop_all(engine)
        # Base_Forecast.metadata.drop_all(engine)

        engine.dispose()


@pytest.fixture()
def db_connection(engine_url):
    database_connection = DatabaseConnection(engine_url, echo=True)

    engine = database_connection.engine
    # connection = engine.connect()
    # transaction = connection.begin()

    # There should be a way to only make the tables once
    # but make sure we remove the data
    database_connection.create_all()
    Base_PV.metadata.create_all(engine)

    yield database_connection

    # transaction.rollback()
    # connection.close()

    Base_PV.metadata.drop_all(engine)
    Base_Forecast.metadata.drop_all(engine)


@pytest.fixture()
def db_session(db_connection, engine_url):
    """Return a sqlalchemy session, which tears down everything properly post-test."""

    with db_connection.get_session() as s:
        s.begin()
        yield s
        s.rollback()


@pytest.fixture
def nwp_data():
    # Load dataset which only contains coordinates, but no data
    ds = xr.open_zarr(
        f"{os.path.dirname(os.path.abspath(__file__))}/data/sample_data/nwp_shell.zarr"
    )

    # Last init time was at least 2 hours ago and hour to 3-hour interval
    t0_datetime_utc = time_before_present(timedelta(hours=2)).floor(timedelta(hours=3))
    ds.init_time.values[:] = pd.date_range(
        t0_datetime_utc - timedelta(hours=3 * (len(ds.init_time) - 1)),
        t0_datetime_utc,
        freq=timedelta(hours=3),
    )

    # This is important to avoid saving errors
    for v in list(ds.coords.keys()):
        if ds.coords[v].dtype == object:
            ds[v].encoding.clear()

    for v in list(ds.variables.keys()):
        if ds[v].dtype == object:
            ds[v].encoding.clear()

    # Add data to dataset
    ds["UKV"] = xr.DataArray(
        np.zeros([len(ds[c]) for c in ds.coords]),
        coords=ds.coords,
    )

    # Add stored attributes to DataArray
    ds.UKV.attrs = ds.attrs["_data_attrs"]
    del ds.attrs["_data_attrs"]

    return ds


@pytest.fixture()
def sat_data():
    # Load dataset which only contains coordinates, but no data
    ds = xr.open_zarr(
        f"{os.path.dirname(os.path.abspath(__file__))}/data/sample_data/non_hrv_shell.zarr"
    )

    # Change times so they lead up to present. Delayed by an hour
    t0_datetime_utc = time_before_present(timedelta(hours=1)).floor(timedelta(minutes=30))
    ds.time.values[:] = pd.date_range(
        t0_datetime_utc - timedelta(minutes=5 * (len(ds.time) - 1)),
        t0_datetime_utc,
        freq=timedelta(minutes=5),
    )

    # Add data to dataset
    ds["data"] = xr.DataArray(
        np.zeros([len(ds[c]) for c in ds.coords]),
        coords=ds.coords,
    )

    # Add stored attributes to DataArray
    ds.data.attrs = ds.attrs["_data_attrs"]
    del ds.attrs["_data_attrs"]

    return ds


@pytest.fixture()
def gsp_yields_and_systems(db_session):
    """Create gsp yields and systems"""

    # GSP data is mostly up to date
    t0_datetime_utc = time_before_present(timedelta(minutes=0)).floor(timedelta(minutes=30))

    # this pv systems has same coordiantes as the first gsp
    gsp_yields = []
    locations = []
    for i in range(1, 11):
        location_sql: LocationSQL = Location(
            gsp_id=i,
            label=f"GSP_{i}",
            installed_capacity_mw=123.0,
        ).to_orm()

        gsp_yield_sqls = []
        # From 2 hours ago to 8.5 hours into future
        for minute in range(-2 * 60, 9 * 60, 30):
            gsp_yield_sql = GSPYield(
                datetime_utc=t0_datetime_utc + timedelta(minutes=minute),
                solar_generation_kw=np.random.randint(low=0, high=1000),
            ).to_orm()
            gsp_yield_sql.location = location_sql
            gsp_yields.append(gsp_yield_sql)
            locations.append(location_sql)

    # add to database
    db_session.add_all(gsp_yields + locations)

    db_session.commit()

    return {
        "gsp_yields": gsp_yields,
        "gs_systems": locations,
    }


@pytest.fixture()
def sample_datamodule():
    dm = DataModule(
        configuration=None,
        batch_size=2,
        num_workers=0,
        prefetch_factor=2,
        train_period=[None, None],
        val_period=[None, None],
        test_period=[None, None],
        block_nwp_and_sat=False,
        batch_dir="tests/data/sample_batches",
    )
    return dm


@pytest.fixture()
def sample_batch(sample_datamodule):
    batch = next(iter(sample_datamodule.train_dataloader()))
    return batch


@pytest.fixture()
def sample_satellite_batch(sample_batch):
    sat_image = sample_batch[BatchKey.satellite_actual]
    return torch.swapaxes(sat_image, 1, 2)


@pytest.fixture()
def model_minutes_kwargs():
    kwargs = dict(
        forecast_minutes=480,
        history_minutes=120,
    )
    return kwargs


@pytest.fixture()
def encoder_model_kwargs():
    # Used to test encoder model on satellite data
    kwargs = dict(
        sequence_length=90 // 5 - 2,
        image_size_pixels=24,
        in_channels=11,
        out_features=128,
    )
    return kwargs


@pytest.fixture()
def multimodal_model_kwargs(model_minutes_kwargs):
    kwargs = dict(
        image_encoder=pvnet.models.multimodal.encoders.encoders3d.DefaultPVNet,
        encoder_out_features=128,
        encoder_kwargs=dict(
            number_of_conv3d_layers=6,
            conv3d_channels=32,
        ),
        include_sat=True,
        include_nwp=True,
        add_image_embedding_channel=True,
        sat_image_size_pixels=24,
        nwp_image_size_pixels=24,
        number_sat_channels=11,
        number_nwp_channels=2,
        output_network=pvnet.models.multimodal.linear_networks.networks.ResFCNet2,
        output_network_kwargs=dict(
            fc_hidden_features=128,
            n_res_blocks=6,
            res_block_layers=2,
            dropout_frac=0.0,
        ),
        embedding_dim=16,
        include_sun=True,
        include_gsp_yield_history=True,
        sat_history_minutes=90,
        nwp_history_minutes=120,
        nwp_forecast_minutes=480,
    )
    kwargs.update(model_minutes_kwargs)
    return kwargs


@pytest.fixture()
def me_latest(db_session):
    metric_values = make_fake_me_latest(session=db_session, model_name="pvnet_v2")
    db_session.add_all(metric_values)
    db_session.commit()
