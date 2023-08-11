from pvnet.app import app
import tempfile
import zarr
import os

from nowcasting_datamodel.models.forecast import (
    ForecastSQL,
    ForecastValueLatestSQL,
    ForecastValueSevenDaysSQL,
    ForecastValueSQL,
)


def test_app(db_session, nwp_data, sat_data, gsp_yields_and_systems, me_latest):
    # Environment variable DB_URL is set in engine_url, which is called by db_session
    # set NWP_ZARR_PATH
    # save nwp_data to temporary file, and set NWP_ZARR_PATH
    # SATELLITE_ZARR_PATH
    # save sat_data to temporary file, and set SATELLITE_ZARR_PATH
    # GSP data

    with tempfile.TemporaryDirectory() as tmpdirname:
        # The app loads sat and NWP data from environment variable
        # Save out data and set paths
        temp_nwp_path = f"{tmpdirname}/nwp.zarr"
        os.environ["NWP_ZARR_PATH"] = temp_nwp_path
        nwp_data.to_zarr(temp_nwp_path)

        # In production sat zarr is zipped
        temp_sat_path = f"{tmpdirname}/sat.zarr.zip"
        os.environ["SATELLITE_ZARR_PATH"] = temp_sat_path
        store = zarr.storage.ZipStore(temp_sat_path, mode="x")
        sat_data.to_zarr(store)
        store.close()

        # Set model version
        os.environ["APP_MODEL_VERSION"] = "96ac8c67fa8663844ddcfa82aece51ef94f34453"

        # Run prediction
        app(gsp_ids=list(range(1, 11)))

    # Check forecasts have been made
    # (10 GSPs + 1 National) = 11 forecasts
    # Doubled for historic and forecast
    forecasts = db_session.query(ForecastSQL).all()
    assert len(forecasts) == 22

    # Check probabilistic added
    assert "90" in forecasts[0].forecast_values[0].properties
    assert "10" in forecasts[0].forecast_values[0].properties

    # 11 GSPs * 16 time steps in forecast
    assert len(db_session.query(ForecastValueSQL).all()) == 11 * 16
    assert len(db_session.query(ForecastValueLatestSQL).all()) == 11 * 16
    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == 11 * 16
