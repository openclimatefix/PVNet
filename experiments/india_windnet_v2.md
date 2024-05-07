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

Comperison  with the production model:

| Timestep | Prod MAE % | No Meteomatics MAE % | Meteomatics MAE % |
| --- | --- | --- | --- |
| 0-0 minutes | 7.586 | 5.920 | 2.475 |
| 15-15 minutes | 8.021 | 5.809 | 2.968 |
| 30-45 minutes | 7.233 | 5.742 | 3.472 |
| 45-60 minutes | 7.187 | 5.698 | 3.804 |
| 60-120 minutes | 7.231 | 5.816 | 4.650 |
| 120-240 minutes | 7.287 | 6.080 | 6.028 |
| 240-360 minutes | 7.319 | 6.375 | 6.738 |
| 360-480 minutes | 7.285 | 6.638 | 6.964 |
| 480-720 minutes | 7.143 | 6.747 | 6.906 |
| 720-1440 minutes | 7.380 | 7.207 | 6.962 |
| 1440-2880 minutes | 7.904 | 7.507 | 7.507 |


Example plot

![Screenshot_20240430_082937](https://github.com/openclimatefix/PVNet/assets/7170359/88db342e-bf82-414e-8255-5ad4af659fb8)
