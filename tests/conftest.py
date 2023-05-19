import os

import pytest
import torch
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.models.base import Base_Forecast, Base_PV
from ocf_datapipes.utils.consts import BatchKey
from testcontainers.postgres import PostgresContainer

import pvnet
from pvnet.data.datamodule import DataModule

# TODO copy over to this repo
from nowcasting_forecast.utils import floor_minutes_dt


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
    # middle of the UK
    x_center_osgb = 500_000
    y_center_osgb = 500_000
    t0_datetime_utc = floor_minutes_dt(datetime.utcnow()) - timedelta(hours=1)
    image_size = 1000
    time_steps = 10

    x, y = make_image_coords_osgb(
        size_x=image_size,
        size_y=image_size,
        x_center_osgb=x_center_osgb,
        y_center_osgb=y_center_osgb,
        km_spacing=2,
    )

    # time = pd.date_range(start=t0_datetime_utc, freq="30T", periods=10)
    step = [timedelta(minutes=60 * i) for i in range(0, time_steps)]

    coords = (
        ("init_time", [t0_datetime_utc]),
        ("variable", np.array(["dswrf", "t", "prate", "si10"])),
        ("step", step),
        ("x", x),
        ("y", y),
    )

    nwp = xr.DataArray(
        abs(  # to make sure average is about 100
            np.random.uniform(
                0,
                200,
                size=(1, 4, time_steps, image_size, image_size),
            )
        ),
        coords=coords,
        name="data",
    )  # Fake data for testing!
    return nwp.to_dataset(name="UKV")


@pytest.fixture()
def sat_data():
    # middle of the UK
    t0_datetime_utc = floor_minutes_dt(datetime.utcnow()) - timedelta(hours=1.5)

    times = [t0_datetime_utc]
    # this means there will be about 30 mins of no data.
    # This reflects the true satellite consumer
    time_steps = 20
    for i in range(1, time_steps):
        times.append(t0_datetime_utc + timedelta(minutes=5 * i))

    local_path = os.path.dirname(os.path.abspath(__file__))
    x, y = np.load(f"{local_path}/sat_data/geo.npy", allow_pickle=True)

    coords = (
        ("time", times),
        ("x_geostationary", x),
        ("y_geostationary", y),
        (
            "variable",
            np.array(
                [
                    "IR_016",
                    "IR_039",
                    "IR_087",
                    "IR_097",
                    "IR_108",
                    "IR_120",
                    "IR_134",
                    "VIS006",
                    "VIS008",
                    "WV_062",
                    "WV_073",
                ]
            ),
        ),
    )

    sat = xr.DataArray(
        abs(  # to make sure average is about 100
            np.random.uniform(
                0,
                200,
                size=(time_steps, 615, 298, 11),
            )
        ),
        coords=coords,
        name="data",
    )  # Fake data for testing!

    area_attr = np.load(f"{local_path}/sat_data/area.npy")
    sat.attrs["area"] = area_attr

    sat["x_osgb"] = sat.x_geostationary
    sat["y_osgb"] = sat.y_geostationary

    return sat.to_dataset(name="data").sortby("time")


@pytest.fixture()
def gsp_yields_and_systems(db_session):
    """Create gsp yields and systems

    gsp systems: One systems
    GSP yields:
        For system 1, gsp yields from 2 hours ago to 8 in the future at 30 minutes intervals
        For system 2: 1 gsp yield at 16.00
    """

    # this pv systems has same coordiantes as the first gsp
    gsp_yield_sqls = []
    locations = []
    for i in range(N_GSP):
        location_sql_1: LocationSQL = Location(
            gsp_id=i + 1,
            label=f"GSP_{i+1}",
            installed_capacity_mw=123.0,
        ).to_orm()

        t0_datetime_utc = floor_minutes_dt(datetime.now(timezone.utc)) - timedelta(hours=2)

        gsp_yield_sqls = []
        for hour in range(0, 10):
            for minute in range(0, 60, 30):
                datetime_utc = t0_datetime_utc + timedelta(hours=hour - 2, minutes=minute)
                gsp_yield_1 = GSPYield(
                    datetime_utc=datetime_utc,
                    solar_generation_kw=20 + hour + minute,
                ).to_orm()
                gsp_yield_1.location = location_sql_1
                gsp_yield_sqls.append(gsp_yield_1)
                locations.append(location_sql_1)

    # add to database
    db_session.add_all(gsp_yield_sqls + locations)

    db_session.commit()

    return {
        "gsp_yields": gsp_yield_sqls,
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
