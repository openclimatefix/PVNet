import wandb
import pandas as pd
import numpy as np
api = wandb.Api()
run = api.run("openclimatefix/india/5llq8iw6")

print(run.history())
df = run.history()
print(df.columns)
# Get the columns that are in the format 'MAE_horizon/step_<number>/val`
mae_cols = [col for col in df.columns if "MAE_horizon/step_" in col and "val" in col]
print(mae_cols)
print(len(mae_cols))
# Sort them
mae_cols.sort()
print(mae_cols)
# Want MAE average of these groupings: [[0],[1],[2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15,16,17,18,19,20], etc]
df = df[mae_cols]
print(df)
# Get last non-NaN value
# Drop all rows with all NaNs
df = df.dropna(how="all")
print(df)
# Select the last row
print(df.iloc[-1])
df = df.iloc[-1]
# Error in the grouping from above
# Want MAE average of these groupings: [[0],[1],[2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15,16,17,18,19,20], etc]
# breakdowns we want, 192 timesteps, each timestep is 15 minutes
# 0-15 minutes
# 15-30 minutes
# 30-60 minutes
# 60-120 minutes
# 120-240 minutes
# 240-360 minutes
# 360-480 minutes
# 480-720 minutes
# 12-24 hours
# 24-48 hours

# Calculate the timedelta for each group
# Get the step from the column name
column_timesteps = [int(col.split("_")[-1].split("/")[0])*15 for col in mae_cols]
print(column_timesteps)
# Get the timedelta for each group
groupings = [[0,0], [15,15], [30,45], [45,60], [60,120], [120,240], [240,360], [360,480], [480,720], [720,1440], [1440,2880]]

run = api.run("openclimatefix/india/v3mja33d")
meteo_df = run.history()
meteo_df = meteo_df[mae_cols]
meteo_df = meteo_df.dropna(how="all")
print(meteo_df)
# Get average across entire row, and get the IDX for the one with the smallest values
min_row_mean = np.inf
for idx, (row_idx, row) in enumerate(meteo_df.iterrows()):
    if row.mean() < min_row_mean:
        min_row_mean = row.mean()
        min_row_idx = idx
print(min_row_idx)
print(meteo_df.iloc[min_row_idx])
meteo_df = meteo_df.iloc[min_row_idx]

print("| Timestep | Prod MAE % | Meteomatics MAE % |")
print("| --- | --- | --- |")
for grouping in groupings:
    # Select indicies from column_timesteps that are within the grouping, inclusive
    group_idx = [idx for idx, timestep in enumerate(column_timesteps) if timestep >= grouping[0] and timestep <= grouping[1]]
    print(f"| {grouping[0]}-{grouping[1]} minutes | {df[group_idx].mean()*100.:0.3f} | {meteo_df[group_idx].mean()*100.:0.3f} |")


