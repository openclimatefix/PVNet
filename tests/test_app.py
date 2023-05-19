from pvnet.app import main
from nowcasting_datamodel.models import ForecastSQL



def test_main(db_session, nwp_data, sat_data, gsp_data):

    # set DB_URL
    # set NWP_ZARR_PATH
    # save nwp_data to temporary file, and set NWP_ZARR_PATH
    # SATELLITE_ZARR_PATH
    # save sat_data to temporary file, and set SATELLITE_ZARR_PATH
    # GSP data

    main()

    # check forecasts are made using db_session
    forecasts = db_session.query(ForecastSQL).all()
    assert len(forecasts) == 10

    assert len(forecasts[0].forecast_values) > 0