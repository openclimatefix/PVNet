"""
Script to generate a table comparing two run for MAE values for 48 hour 15 minute forecast
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb


def main(runs: list[str], run_names: list[str]) -> None:
    """
    Compare two runs for MAE values for 48 hour 15 minute forecast
    """
    api = wandb.Api()
    dfs = []
    for run in runs:
        run = api.run(f"openclimatefix/india/{run}")

        df = run.history()
        # Get the columns that are in the format 'MAE_horizon/step_<number>/val`
        mae_cols = [col for col in df.columns if "MAE_horizon/step_" in col and "val" in col]
        # Sort them
        mae_cols.sort()
        df = df[mae_cols]
        # Get last non-NaN value
        # Drop all rows with all NaNs
        df = df.dropna(how="all")
        # Select the last row
        # Get average across entire row, and get the IDX for the one with the smallest values
        min_row_mean = np.inf
        for idx, (row_idx, row) in enumerate(df.iterrows()):
            if row.mean() < min_row_mean:
                min_row_mean = row.mean()
                min_row_idx = idx
        df = df.iloc[min_row_idx]
        # Calculate the timedelta for each group
        # Get the step from the column name
        column_timesteps = [int(col.split("_")[-1].split("/")[0]) * 15 for col in mae_cols]
        dfs.append(df)
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

    groups_df = []
    grouping_starts = [grouping[0] for grouping in groupings]
    header = "| Timestep |"
    separator = "| --- |"
    for run_name in run_names:
        header += f" {run_name} MAE % |"
        separator += " --- |"
    print(header)
    print(separator)
    for grouping in groupings:
        group_string = f"| {grouping[0]}-{grouping[1]} minutes |"
        # Select indicies from column_timesteps that are within the grouping, inclusive
        group_idx = [
            idx
            for idx, timestep in enumerate(column_timesteps)
            if timestep >= grouping[0] and timestep <= grouping[1]
        ]
        data_one_group = []
        for df in dfs:
            mean_row = df.iloc[group_idx].mean()
            group_string += f" {mean_row:0.3f} |"
            data_one_group.append(mean_row)
        print(group_string)

        groups_df.append(data_one_group)

    groups_df = pd.DataFrame(groups_df, columns=run_names, index=grouping_starts)

    for idx, df in enumerate(dfs):
        print(f"{run_names[idx]}: {df.mean()*100:0.3f}")

    # Plot the error on per timestep, and all timesteps
    plt.figure()
    for idx, df in enumerate(dfs):
        plt.plot(column_timesteps, df, label=run_names[idx])
    plt.legend()
    plt.xlabel("Timestep (minutes)")
    plt.ylabel("MAE %")
    plt.title("MAE % for each timestep")
    plt.savefig("mae_per_timestep.png")
    plt.show()

    # Plot the error on per timestep, and grouped timesteps
    plt.figure()
    for run_name in run_names:
        plt.plot(groups_df[run_name], label=run_name)
    plt.legend()
    plt.xlabel("Timestep (minutes)")
    plt.ylabel("MAE %")
    plt.title("MAE % for each timestep")
    plt.savefig("mae_per_timestep.png")
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    "5llq8iw6"
    parser.add_argument("--first_run", type=str, default="xdlew7ib")
    parser.add_argument("--second_run", type=str, default="v3mja33d")
    # Add arguments that is a list of strings
    parser.add_argument("--list_of_runs", nargs="+")
    parser.add_argument("--run_names", nargs="+")
    args = parser.parse_args()
    main(args.list_of_runs, args.run_names)
