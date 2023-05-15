"""App to run inference"""

# imports



# logging set up

# ENVIORNMENT VARIABLES
# NWP_ZARR_PATH = s3://nowcasting-nwp-development/data/latest.zarr
# SATELLITE_ZARR_PATH = s3://nowcasting-sat-development/data/latest/latest.zarr.zip
# DB_URL, ssh tunnel # later


# main app
# click arguments
# config file, should use DB_URL, NWP_ZARR_PATH, SATELLITE_ZARR_PATH
# inference datetime can be None
def app():

    # 0. If inference datetime is None, round down to last 30 minutes

    # 1. set up datapipe line
    # check we get all gsp_ids
    # https://github.com/openclimatefix/ocf_datapipes/blob/main/ocf_datapipes/production/power_perceiver.py

    # 2. set up model (loading it for HF)

    # 3. run through the batches

    # 4. merge batche results to pandas df
    # columns should be
    # - target_datetime_utc
    # - gsp_id
    # - expected_power_generation_megawatts

    # 5. filter for sun angle
    # PD: add example of code

    # 6. add up to make national total
    # add gsp_id -=0  in dataframe

    # 7. convert to ForecastSQL object
    # maybe PD

    # 8. write to database, using general save function
    # maybe PD
    # https://github.com/openclimatefix/nowcasting_datamodel/blob/main/nowcasting_datamodel/save/save.py#L19