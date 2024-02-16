# PVNet for Wind and PV Sites in India

## PVNet for sites

### Data

We use PV generation data for India from April 2019-Nov 2022 for training
and Dec 2022- Nov 2023 for validation. This is only with ECMWF data, and PV generation history.

The forecast is every 15 minutes for 48 hours for PV generation.

The input NWP data is hourly, and 32x32 pixels (corresponding to around 320kmx320km) around a central
point in NW-India.

[WandB Link](https://wandb.ai/openclimatefix/pvnet_india2.1/runs/o4xpvzrc)

### Results

Overall MAE is 4.9% on the validation set, and forecasts look overall good.

![batch_idx_1_all_892_2ca7e12db5de2cf2e244](https://github.com/openclimatefix/PVNet/assets/7170359/07e8199a-11b5-4400-9897-37b7738a4f39)

![W B Chart 05_02_2024, 10_07_12_pvnet](https://github.com/openclimatefix/PVNet/assets/7170359/abaefdc1-dedd-4a12-8a26-afaf36d7786b)

## WindNet


### Data

We use Wind generation data for India from April 2019-Nov 2022 for training
and Dec 2022- Nov 2023 for validation. This is only with ECMWF data, and Wind generation history.

The forecast is every 15 minutes for 48 hours for Wind generation.

The input NWP data is hourly, and 32x32 pixels (corresponding to around 320kmx320km) around a central
point in NW-India. Note: The majority of the wind generation is likely not covered in the 320kmx320km area.


[WandB Link](https://wandb.ai/openclimatefix/pvnet_india2.1/runs/otdx7axx)

### Results

![W B Chart 05_02_2024, 10_05_19](https://github.com/openclimatefix/PVNet/assets/7170359/6a8cd9c5-bdfe-41ab-996d-37fd1be2a07c)

![W B Chart 05_02_2024, 10_06_51_windnet](https://github.com/openclimatefix/PVNet/assets/7170359/77554ef0-4411-4432-af95-8530aef4a701)

![batch_idx_1_all_1730_379a9f881a7f01153f98](https://github.com/openclimatefix/PVNet/assets/7170359/243d9f3e-4cb9-405e-80c5-40c6c218c17f)

MAE is around 10% overall, although it doesn't seem to do very well on the ramps up and down.
