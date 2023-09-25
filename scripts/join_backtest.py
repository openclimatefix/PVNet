""" A script to download and processing the PVNet backtests that have been uploaded to GCP.

To ouput the forecast as a zarr.
"""

import click
import fsspec
import xarray as xr


@click.command()
@click.option(
    "--dir",
    prompt="Directory to process",
    type=click.STRING,
    help="The directory to process.",
)
@click.option(
    "--save_to",
    prompt="Location to save to with filename",
    type=click.Path(),
    help="The location to save the processed data including the filename.",
)
def main(dir, save_to):
    """
    Process the specified directory and save the processed data to the specified location.
    """
    # get all the files
    fs = fsspec.open(dir).fs
    files = fs.ls(dir)

    # Can select a proportion of the data to processes as a test
    # N_start = 10080
    # N_end = 10100

    N_files = len(files)
    all_dataset_xr = []
    # Change to iterate through N_start to N_end if wanting to use a sample
    for i, filename in enumerate(files):
        print(f"{round(i/N_files*100)}%")

        ## get all files in a directory
        with fsspec.open(f"gs://{filename}", mode="rb") as file:
            dataset = xr.open_dataset(file, engine="h5netcdf")
            national = dataset.sel(gsp_id=0)
            national = national.assign_coords(
                forecast_init_time=national.forecast_init_time.values
            )
            idx = range(0, len(national.target_datetime_utc.values))
            national = national.assign_coords(target_datetime_utc=idx)
            national = national.load()
            all_dataset_xr.append(national)

    print("Merging data")
    all_dataset_xr = xr.concat(all_dataset_xr, dim="forecast_init_time")
    print("Data merged, now saving")
    all_dataset_xr.to_zarr(save_to)
    print(f"Saved Zarr to {save_to}")


if __name__ == "__main__":
    main()
