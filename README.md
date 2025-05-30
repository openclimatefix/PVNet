# PVNet 2.1
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-19-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

 [![Python Bump Version & release](https://github.com/openclimatefix/PVNet/actions/workflows/release.yml/badge.svg)](https://github.com/openclimatefix/PVNet/actions/workflows/release.yml) [![ease of contribution: hard](https://img.shields.io/badge/ease%20of%20contribution:%20hard-bb2629)](https://github.com/openclimatefix/ocf-meta-repo?tab=readme-ov-file#overview-of-ocfs-nowcasting-repositories)


This project is used for training PVNet and running PVNet on live data.

PVNet2 is a multi-modal late-fusion model that largely inherits the same architecture from
[PVNet1.0](https://github.com/openclimatefix/predict_pv_yield). The NWP (Numerical Weather Prediction) and
satellite data are sent through some neural network which encodes them down to
1D intermediate representations. These are concatenated together with the GSP (Grid Supply Point)
output history, the calculated solar coordinates (azimuth and elevation) and the
GSP ID which has been put through an embedding layer. This 1D concatenated
feature vector is put through an output network which outputs predictions of the
future GSP yield. National forecasts are made by adding all the GSP forecasts
together.


## Experiments

Our paper based on this repo was accepted into the Tackling Climate Change with Machine Learning workshop at ICLR 2024 and can be viewed [here](https://www.climatechange.ai/papers/iclr2024/46).

Some slightly more structured notes on deliberate experiments we have performed with PVNet are [here](https://docs.google.com/document/d/1VumDwWd8YAfvXbOtJEv3ZJm_FHQDzrKXR0jU9vnvGQg).

Some very rough, early working notes on this model are
[here](https://docs.google.com/document/d/1fbkfkBzp16WbnCg7RDuRDvgzInA6XQu3xh4NCjV-WDA). These are now somewhat out of date.



## Setup / Installation

```bash
git clone https://github.com/openclimatefix/PVNet.git
cd PVNet
pip install .
```

The commit history is extensive. To save download time, use a depth of 1:
```bash
git clone --depth 1 https://github.com/openclimatefix/PVNet.git
```
This means only the latest commit and its associated files will be downloaded.

Next, in the PVNet repo, install PVNet as an editable package:

```bash
pip install -e .
```

### Additional development dependencies

```bash
pip install ".[dev]"
```



## Getting started with running PVNet

Before running any code in PVNet, copy the example configuration to a
configs directory:

```
cp -r configs.example configs
```

You will be making local amendments to these configs. See the README in
`configs.example` for more info.

### Datasets

As a minimum, in order to create batches of data/run PVNet, you will need to
supply paths to NWP and GSP data. PV data can also be used. We list some
suggested locations for downloading such datasets below:

**GSP (Grid Supply Point)** - Regional PV generation data\
The University of Sheffield provides API access to download this data:
https://www.solar.sheffield.ac.uk/api/

Documentation for querying generation data aggregated by GSP region can be found
here:
https://docs.google.com/document/d/e/2PACX-1vSDFb-6dJ2kIFZnsl-pBQvcH4inNQCA4lYL9cwo80bEHQeTK8fONLOgDf6Wm4ze_fxonqK3EVBVoAIz/pub#h.9d97iox3wzmd

**NWP (Numerical weather predictions)**\
OCF maintains a Zarr formatted version of the German Weather Service's (DWD)
ICON-EU NWP model here:
https://huggingface.co/datasets/openclimatefix/dwd-icon-eu which includes the UK

**PV**\
OCF maintains a dataset of PV generation from 1311 private PV installations
here: https://huggingface.co/datasets/openclimatefix/uk_pv


### Connecting with ocf-data-sampler for batch creation

Outside the PVNet repo, clone the ocf-data-sampler repo and exit the conda env created for PVNet: https://github.com/openclimatefix/ocf-data-sampler
```bash
git clone https://github.com/openclimatefix/ocf-data-sampler.git
conda create -n ocf-data-sampler python=3.11
```

Then go inside the ocf-data-sampler repo to add packages

```bash
pip install .
```

Then exit this environment, and enter back into the pvnet conda environment and install ocf-data-sampler in editable mode (-e). This means the package is directly linked to the source code in the ocf-data-sampler repo.

```bash
pip install -e <PATH-TO-ocf-data-sampler-REPO>
```

If you install the local version of `ocf-data-sampler` that is more recent than the version specified in PVNet, you might receive a warning. However, it should still function correctly.

## Generating pre-made batches of data for training/validation of PVNet

PVNet contains a script for generating batches of data suitable for training the PVNet models. To run the script you will need to make some modifications to the datamodule configuration.

Make sure you have copied the example configs (as already stated above):
```
cp -r configs.example configs
```

### Set up and config example for batch creation

We will use the following example config file for creating batches: `/PVNet/configs/datamodule/configuration/example_configuration.yaml`. Ensure that the file paths are set to the correct locations in `example_configuration.yaml`: search for `PLACEHOLDER` to find where to input the location of the files. You will need to comment out or delete the parts of `example_configuration.yaml` pertaining to the data you are not using.


When creating batches, an additional datamodule config located in `PVNet/configs/datamodule` is passed into the batch creation script: `streamed_batches.yaml`. Like before, a placeholder variable is used when specifying which configuration to use:

```yaml
configuration: "PLACEHOLDER.yaml"
```

This should be given the whole path to the config on your local machine, for example:

```yaml
configuration: "/FULL-PATH-TO-REPO/PVNet/configs/datamodule/configuration/example_configuration.yaml"
```

Where `FULL-PATH-TO-REPO` represent the whole path to the PVNet repo on your local machine.

This is also where you can update the train, val & test periods to cover the data you have access to.

### Running the batch creation script

Run the `save_samples.py` script to create batches with the parameters specified in the datamodule config (`streamed_batches.yaml` in this example):

```bash
python scripts/save_samples.py
```
PVNet uses
[hydra](https://hydra.cc/) which enables us to pass variables via the command
line that will override the configuration defined in the `./configs` directory, like this:

```bash
python scripts/save_samples.py datamodule=streamed_batches datamodule.sample_output_dir="./output" datamodule.num_train_batches=10 datamodule.num_val_batches=5
```

`scripts/save_samples.py` needs a config under `PVNet/configs/datamodule`. You can adapt `streamed_batches.yaml` or create your own in the same folder.

If downloading private data from a GCP bucket make sure to authenticate gcloud (the public satellite data does not need authentication):

```
gcloud auth login
```

Files stored in multiple locations can be added as a list. For example, in the `example_configuration.yaml` file we can supply a path to satellite data stored on a bucket:

```yaml
satellite:
    satellite_zarr_path: gs://solar-pv-nowcasting-data/satellite/EUMETSAT/SEVIRI_RSS/v4/2020_nonhrv.zarr
```

Or to satellite data hosted by Google:

```yaml
satellite:
    satellite_zarr_paths:
      - "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v4/2020_nonhrv.zarr"
      - "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v4/2021_nonhrv.zarr"
```

ocf-data-sampler is currently set up to use 11 channels from the satellite data, the 12th of which is HRV and is not included in these.


### Training PVNet

How PVNet is run is determined by the extensive configuration in the config
files. The configs stored in `PVNet/configs.example` should work with batches created using the steps and batch creation config mentioned above.

Make sure to update the following config files before training your model:

1. In `configs/datamodule/local_premade_batches.yaml`:
    - update `batch_dir` to point to the directory you stored your batches in during batch creation
2. In `configs/model/local_multimodal.yaml`:
    - update the list of encoders to reflect the data sources you are using. If you are using different NWP sources, the encoders for these should follow the same structure with two important updates:
        - `in_channels`: number of variables your NWP source supplies
        - `image_size_pixels`: spatial crop of your NWP data. It depends on the spatial resolution of your NWP; should match `image_size_pixels_height` and/or `image_size_pixels_width` in `datamodule/configuration/site_example_configuration.yaml` for the NWP, unless transformations such as coarsening was applied (e. g. as for ECMWF data)
3. In `configs/local_trainer.yaml`:
    - set `accelerator: 0` if running on a system without a supported GPU

If creating copies of the config files instead of modifying existing ones, update `defaults` in the main `./configs/config.yaml` file to use
your customised config files:

```yaml
defaults:
  - trainer: local_trainer.yaml
  - model: local_multimodal.yaml
  - datamodule: local_premade_batches.yaml
  - callbacks: null
  - logger: csv.yaml
  - experiment: null
  - hparams_search: null
  - hydra: default.yaml
```

Assuming you ran the `save_samples.py` script to generate some premade train and
val data batches, you can now train PVNet by running:

```
python run.py
```

## Backtest

If you have successfully trained a PVNet model and have a saved model checkpoint you can create a backtest using this, e.g. forecasts on historical data to evaluate forecast accuracy/skill. This can be done by running one of the scripts in this repo such as [the UK GSP backtest script](scripts/backtest_uk_gsp.py) or the [the pv site backtest script](scripts/backtest_sites.py), further info on how to run these are in each backtest file.


## Testing

You can use `python -m pytest tests` to run tests

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/felix-e-h-p"><img src="https://avatars.githubusercontent.com/u/137530077?v=4?s=100" width="100px;" alt="Felix"/><br /><sub><b>Felix</b></sub></a><br /><a href="https://github.com/openclimatefix/PVNet/commits?author=felix-e-h-p" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Sukh-P"><img src="https://avatars.githubusercontent.com/u/42407101?v=4?s=100" width="100px;" alt="Sukhil Patel"/><br /><sub><b>Sukhil Patel</b></sub></a><br /><a href="https://github.com/openclimatefix/PVNet/commits?author=Sukh-P" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dfulu"><img src="https://avatars.githubusercontent.com/u/41546094?v=4?s=100" width="100px;" alt="James Fulton"/><br /><sub><b>James Fulton</b></sub></a><br /><a href="https://github.com/openclimatefix/PVNet/commits?author=dfulu" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AUdaltsova"><img src="https://avatars.githubusercontent.com/u/43303448?v=4?s=100" width="100px;" alt="Alexandra Udaltsova"/><br /><sub><b>Alexandra Udaltsova</b></sub></a><br /><a href="https://github.com/openclimatefix/PVNet/commits?author=AUdaltsova" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/zakwatts"><img src="https://avatars.githubusercontent.com/u/47150349?v=4?s=100" width="100px;" alt="Megawattz"/><br /><sub><b>Megawattz</b></sub></a><br /><a href="https://github.com/openclimatefix/PVNet/commits?author=zakwatts" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/peterdudfield"><img src="https://avatars.githubusercontent.com/u/34686298?v=4?s=100" width="100px;" alt="Peter Dudfield"/><br /><sub><b>Peter Dudfield</b></sub></a><br /><a href="https://github.com/openclimatefix/PVNet/commits?author=peterdudfield" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mahdilamb"><img src="https://avatars.githubusercontent.com/u/4696915?v=4?s=100" width="100px;" alt="Mahdi Lamb"/><br /><sub><b>Mahdi Lamb</b></sub></a><br /><a href="#infra-mahdilamb" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://www.jacobbieker.com"><img src="https://avatars.githubusercontent.com/u/7170359?v=4?s=100" width="100px;" alt="Jacob Prince-Bieker"/><br /><sub><b>Jacob Prince-Bieker</b></sub></a><br /><a href="https://github.com/openclimatefix/PVNet/commits?author=jacobbieker" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/codderrrrr"><img src="https://avatars.githubusercontent.com/u/149995852?v=4?s=100" width="100px;" alt="codderrrrr"/><br /><sub><b>codderrrrr</b></sub></a><br /><a href="https://github.com/openclimatefix/PVNet/commits?author=codderrrrr" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://chrisxbriggs.com"><img src="https://avatars.githubusercontent.com/u/617309?v=4?s=100" width="100px;" alt="Chris Briggs"/><br /><sub><b>Chris Briggs</b></sub></a><br /><a href="https://github.com/openclimatefix/PVNet/commits?author=confusedmatrix" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/tmi"><img src="https://avatars.githubusercontent.com/u/147159?v=4?s=100" width="100px;" alt="tmi"/><br /><sub><b>tmi</b></sub></a><br /><a href="https://github.com/openclimatefix/PVNet/commits?author=tmi" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://rdrn.me/"><img src="https://avatars.githubusercontent.com/u/19817302?v=4?s=100" width="100px;" alt="Chris Arderne"/><br /><sub><b>Chris Arderne</b></sub></a><br /><a href="https://github.com/openclimatefix/PVNet/commits?author=carderne" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Dakshbir"><img src="https://avatars.githubusercontent.com/u/144359831?v=4?s=100" width="100px;" alt="Dakshbir"/><br /><sub><b>Dakshbir</b></sub></a><br /><a href="https://github.com/openclimatefix/PVNet/commits?author=Dakshbir" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/MAYANK12SHARMA"><img src="https://avatars.githubusercontent.com/u/145884197?v=4?s=100" width="100px;" alt="MAYANK SHARMA"/><br /><sub><b>MAYANK SHARMA</b></sub></a><br /><a href="https://github.com/openclimatefix/PVNet/commits?author=MAYANK12SHARMA" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/lambaaryan011"><img src="https://avatars.githubusercontent.com/u/153702847?v=4?s=100" width="100px;" alt="aryan lamba "/><br /><sub><b>aryan lamba </b></sub></a><br /><a href="https://github.com/openclimatefix/PVNet/commits?author=lambaaryan011" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/michael-gendy"><img src="https://avatars.githubusercontent.com/u/64384201?v=4?s=100" width="100px;" alt="michael-gendy"/><br /><sub><b>michael-gendy</b></sub></a><br /><a href="https://github.com/openclimatefix/PVNet/commits?author=michael-gendy" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://adityasuthar.github.io/"><img src="https://avatars.githubusercontent.com/u/95685363?v=4?s=100" width="100px;" alt="Aditya Suthar"/><br /><sub><b>Aditya Suthar</b></sub></a><br /><a href="https://github.com/openclimatefix/PVNet/commits?author=adityasuthar" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/markus-kreft"><img src="https://avatars.githubusercontent.com/u/129367085?v=4?s=100" width="100px;" alt="Markus Kreft"/><br /><sub><b>Markus Kreft</b></sub></a><br /><a href="https://github.com/openclimatefix/PVNet/commits?author=markus-kreft" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://jack-kelly.com"><img src="https://avatars.githubusercontent.com/u/460756?v=4?s=100" width="100px;" alt="Jack Kelly"/><br /><sub><b>Jack Kelly</b></sub></a><br /><a href="#ideas-JackKelly" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
