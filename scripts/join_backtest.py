""" A script to download and processing the PVNet backtests that have been uploaded to GCP,
    to ouput the pv data saved as Zarr. Currently set up for the 2022 backtest used for
    the PVLive and PVNet evaluation.
"""
import xarray as xr
import fsspec

dir = "gs://solar-pv-nowcasting-data/backtest/pvnet_v2_2022/hindcasts/"
# filename = 'gs://solar-pv-nowcasting-data/backtest/pvnet_v2_2022/hindcasts/2022-01-01T03:00:00.nc'

# get all the files
fs = fsspec.open(dir).fs
files = fs.ls(dir)


N_start = 10080
# N_end = 10580
N_end = 10100
N = N_end - N_start
N_files = len(files)
all_dataset_xr = []
for i, filename in enumerate(files):  # [N_start:N_end]):
    print(f"{round(i/N_files*100)}%")

    ## get all files in a directory
    with fsspec.open(f"gs://{filename}", mode="rb") as file:
        dataset = xr.open_dataset(file, engine="h5netcdf")

        # just select national
        national = dataset.sel(gsp_id=0)

        # assign forecast_init_time as coordinate
        national = national.assign_coords(
            forecast_init_time=national.forecast_init_time.values
        )

        # drop target_time
        idx = range(0, len(national.target_datetime_utc.values))
        national = national.assign_coords(target_datetime_utc=idx)

        # load the data
        national = national.load()

        all_dataset_xr.append(national)

print(all_dataset_xr[0])

print("next")

# print(all_dataset_xr)

# join datasets together
all_dataset_xr = xr.concat(all_dataset_xr, dim="forecast_init_time")

print("PROCESSED")

print(all_dataset_xr)

# dnetcdf = all_dataset_xr.to_netcdf()

# print(dnetcdf)

all_dataset_xr.to_zarr("/home/zak/data/fc_bt_comp/pvnet_backtest_2022.zarr")
