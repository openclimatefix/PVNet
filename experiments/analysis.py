"""
Script to generate a table comparing two run for MAE values for 48 hour 15 minute forecast
"""
import numpy as np
import wandb
import argparse


def main(first_run: str, second_run: str) -> None:
    """
    Compare two runs for MAE values for 48 hour 15 minute forecast
    """
    api = wandb.Api()
    run = api.run(f"openclimatefix/india/{first_run}")

    df = run.history()
    # Get the columns that are in the format 'MAE_horizon/step_<number>/val`
    mae_cols = [col for col in df.columns if "MAE_horizon/step_" in col and "val" in col]
    # Sort them
    mae_cols.sort()
    # Want MAE average of these groupings: [[0],[1],[2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15,16,17,18,19,20], etc]
    df = df[mae_cols]
    # Get last non-NaN value
    # Drop all rows with all NaNs
    df = df.dropna(how="all")
    # Select the last row
    df = df.iloc[-1]
    # Calculate the timedelta for each group
    # Get the step from the column name
    column_timesteps = [int(col.split("_")[-1].split("/")[0]) * 15 for col in mae_cols]
    # Get the timedelta for each group
    groupings = [
        [0, 0],
        [15, 15],
        [30, 45],
        [45, 60],
        [60, 120],
        [120, 240],
        [240, 360],
        [360, 480],
        [480, 720],
        [720, 1440],
        [1440, 2880],
    ]

    run = api.run(f"openclimatefix/india/{second_run}")
    meteo_df = run.history()
    meteo_df = meteo_df[mae_cols]
    meteo_df = meteo_df.dropna(how="all")
    # Get average across entire row, and get the IDX for the one with the smallest values
    min_row_mean = np.inf
    for idx, (row_idx, row) in enumerate(meteo_df.iterrows()):
        if row.mean() < min_row_mean:
            min_row_mean = row.mean()
            min_row_idx = idx
    meteo_df = meteo_df.iloc[min_row_idx]

    print("| Timestep | First Model MAE % | Second Model MAE % |")
    print("| --- | --- | --- |")
    for grouping in groupings:
        # Select indicies from column_timesteps that are within the grouping, inclusive
        group_idx = [
            idx
            for idx, timestep in enumerate(column_timesteps)
            if timestep >= grouping[0] and timestep <= grouping[1]
        ]
        print(
            f"| {grouping[0]}-{grouping[1]} minutes | {df[group_idx].mean()*100.:0.3f} | {meteo_df[group_idx].mean()*100.:0.3f} |"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("first_run", type=str, default="5llq8iw6")
    parser.add_argument("second_run", type=str, default="v3mja33d")
    args = parser.parse_args()
    main(args.first_run, args.second_run)