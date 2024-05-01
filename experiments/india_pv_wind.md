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

### WindNet v2 Meteomatics + ECMWF Model

[WandB Linl](https://wandb.ai/openclimatefix/india/runs/v3mja33d)

This newest experiment uses Meteomatics data in addition to ECMWF data. The Meteomatics data is at specific locations corresponding
to the gneeration sites we know about. It is smartly downscaled ECMWF data, down to 15 minutes and at a few height levels we are
interested in, primarily 10m, 100m, and 200m. The Meteomatics data is a semi-reanalysis, with each block of 6 hours being from one forecast run.
For example, in one day, hours 00-06 are from the same, 00 forecast run, and hours 06-12 are from the 06 forecast run. This is important to note
as it is both not a real reanalysis, but we also can't have it exactly match the live data, as any forecast steps beyond 6 hours are thrown away.
This does mean that these results should be taken as a best case or better than best case scenario, as every 6 hour, observations from the future
are incorporated into the Meteomatics input data from the next NWP mode run.

For the purposes of WindNet, Meteomatics data is treated as Sensor data that goes into the future.
The model encodes the sensor information the same way as for the historical PV, Wind, and GSP generation, and has
a simple, single attention head to encode the information. This is then concatenated along with the rest of the data, like in
previous experiments.

This model also has an even larger input size of ECMWF data, 81x81 pixels, corresponding to around 810kmx810km.
![Screenshot_20240430_082855](https://github.com/openclimatefix/PVNet/assets/7170359/6981a088-8664-474b-bfea-c94c777fc119)

MAE is 7.0% on the validation set, showing a slight improvement over the previous model.

Example plot

![Screenshot_20240430_082937](https://github.com/openclimatefix/PVNet/assets/7170359/88db342e-bf82-414e-8255-5ad4af659fb8)

### April-29-2024 WindNet v1 Production Model

[WandB Link](https://wandb.ai/openclimatefix/india/runs/5llq8iw6)

Improvements: Larger input size (64x64), 7 hour delay for ECMWF NWP inputs, to match productions.
New, much more efficient encoder for NWP, allowing for more filters and layers, with less parameters.
The 64x64 input size corresponds to 6.4 degrees x 6.4 degrees, which is around 700km x 700km. This allows for the
model to see the wind over the wind generation sites, which seems to be the biggest reason for the improvement in the model.



MAE is 7.6% with real improvements on the production side of things.


There were other experiments with slightly different numbers of filters, model parameters and the like, but generally no
improvements were seen.


## WindNet v1 Results

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
