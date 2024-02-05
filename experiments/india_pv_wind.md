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


## WindNet


### Data

We use Wind generation data for India from April 2019-Nov 2022 for training
and Dec 2022- Nov 2023 for validation. This is only with ECMWF data, and Wind generation history.

The forecast is every 15 minutes for 48 hours for Wind generation.

The input NWP data is hourly, and 32x32 pixels (corresponding to around 320kmx320km) around a central
point in NW-India.

[WandB Link](https://wandb.ai/openclimatefix/pvnet_india2.1/runs/otdx7axx)

### Results

MAE is around 10% overall, although it doesn't seem to do very well on the ramps up and down.