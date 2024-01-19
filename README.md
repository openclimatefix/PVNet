# PVNet 2.1

[![test-release](https://github.com/openclimatefix/PVNet/actions/workflows/test-release.yml/badge.svg)](https://github.com/openclimatefix/PVNet/actions/workflows/test-release.yml)

This project is used for training PVNet and running PVnet on live data.

PVNet2 is a multi-modal late-fusion model that largely inherits the same architecture from
[PVNet1.0](https://github.com/openclimatefix/predict_pv_yield). The NWP and
satellite data are sent through some neural network which encodes them down to
1D intermediate representations. These are concatenated together with the GSP
output history, the calculated solar coordinates (azimuth and elevation) and the
GSP ID which has been put through an embedding layer. This 1D concatenated
feature vector is put through an output network which outputs predictions of the
future GSP yield. National forecasts are made by adding all the GSP forecasts
together.


## Experiments

Some quite rough working notes on experiments training this model and running it in production are
[here](https://docs.google.com/document/d/1fbkfkBzp16WbnCg7RDuRDvgzInA6XQu3xh4NCjV-WDA/edit?usp=sharing).

## Setup / Installation

```bash
git clone https://github.com/openclimatefix/PVNet.git
cd PVNet
pip install -r requirements.txt
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
pip install -r requirements-dev.txt
```



## Getting started with running PVNet

Before running any code in within PVNet, copy the example configuration to a
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
https://www.solar.sheffield.ac.uk/pvlive/api/

Documentation for querying generation data aggregated by GSP region can be found
here:
https://docs.google.com/document/d/e/2PACX-1vSDFb-6dJ2kIFZnsl-pBQvcH4inNQCA4lYL9cwo80bEHQeTK8fONLOgDf6Wm4ze_fxonqK3EVBVoAIz/pub#h.9d97iox3wzmd

**NWP (Numerical weather predictions)**\
OCF maintains a Zarr formatted version the German Weather Service's (DWD)
ICON-EU NWP model here:
https://huggingface.co/datasets/openclimatefix/dwd-icon-eu which includes the UK

**PV**\
OCF maintains a dataset of PV generation from 1311 private PV installations
here: https://huggingface.co/datasets/openclimatefix/uk_pv


### Connecting with ocf_datapipes for batch creation

Outside the PVNet repo, clone the ocf-datapipes repo and exit the conda env created for PVNet: https://github.com/openclimatefix/ocf_datapipes
```bash
git clone --depth 1 https://github.com/openclimatefix/ocf_datapipes.git
conda create -n ocf_datapipes python=3.10
```

Then go inside the ocf_datapipes repo to add packages

```bash
pip install -r requirements.txt requirements-dev.txt
```

Then exit this environment, and enter back into the pvnet conda environment and install ocf_datapies in editable mode (-e). This means the package is directly linked to the source code in the ocf_datapies repo.

```bash
pip install -e <PATH-TO-ocf_datapipes-REPO>
```

## Generating pre-made batches of data for training/validation of PVNet

PVNet contains a script for generating batches of data suitable for training the PVNet models. To run the script you will need to make some modifications to the datamodule configuration.

Make sure you have copied the example configs (as already stated above):
```
cp -r configs.example configs
```

### Set up and config example for batch creation

We will use the example of creating batches using data from gcp:
`/PVNet/configs/datamodule/configuration/gcp_configuration.yaml`
Ensure that the file paths are set to the correct locations in
`gcp_configuration.yaml`.

`PLACEHOLDER` is used to indcate where to input the location of the files.

For OCF use cases, file locations can be found in `template_configuration.yaml` located alongside `gcp_configuration.yaml`.

In these configurations you can update the train, val & test periods to cover the data you have access to.


With your configuration in place, you can proceed to create batches. PVNet uses
[hydra](https://hydra.cc/) which enables us to pass variables via the command
line that will override the configuration defined in the `./configs` directory.

When creating batches, an additional config is used which is passed into the batch creation script. This is the datamodule config located `PVNet/configs/datamodule`.

For this example we will be using the `streamed_batches.yaml` config. Like before, a placeholder variable is used when specifing which configuration to use:

`configuration: "PLACEHOLDER.yaml"`

This should be given the whole path to the config on your local machine, such as for our example it should be changed to:

`configuration: "/FULL-PATH-TO-REPO/PVNet/configs/datamodule/configuration/gcp_configuration.yaml"
`

Where `FULL-PATH-TO-REPO` represent the whole path to the PVNet repo on your local machine.

### Running the batch creation script

Run the save_batches.py script to create batches with the following example arguments as:

```
python scripts/save_batches.py datamodule=streamed_batches +batch_output_dir="./output" +num_train_batches=10 +num_val_batches=5
```

In this function the datamodule argument looks for a config under `PVNet/configs/datamodule`. The examples here are either to use "premade_batches" or "streamed_batches".

Its important that the dates set for the training, validation and testing in the datamodule (`streamed_batches.yaml`) config are within the ranges of the dates set for the input features in the configuration (`gcp_configuration.yaml`).

If downloading private data from a gcp bucket make sure to authenticate gcloud (the public satellite data does not need authentication):

```
gcloud auth login
```

For files stored in multiple locations they can be added as list. For example from the gcp_configuration.yaml file we can change from satellite data stored on a bucket:

```
satellite:
    satellite_zarr_path: gs://solar-pv-nowcasting-data/satellite/EUMETSAT/SEVIRI_RSS/v4/2020_nonhrv.zarr
```

To satellite data hosted by Google:

```
satellite:
    satellite_zarr_paths:
      - "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v4/2020_nonhrv.zarr"
      - "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v4/2021_nonhrv.zarr"
```
Datapipes is currently set up to use 11 channels from the satellite data, the 12th of which is HRV and is not included in these.


### Training PVNet

How PVNet is run is determined by the extensive configuration in the config
files. The following configs have been tested to work using batches of data
created using the steps and batch creation config mentioned above.

You should create the following configs before trying to train a model locally,
as so:

In `configs/datamodule/local_premade_batches.yaml`:

```yaml
_target_: pvnet.data.datamodule.DataModule
configuration: null
batch_dir: "./output" # where the batches are saved
num_workers: 20
prefetch_factor: 2
batch_size: 8
```

In `configs/model/local_multimodal.yaml`:

```yaml
_target_: pvnet.models.multimodal.multimodal.Model

output_quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]

#--------------------------------------------
# NWP encoder
#--------------------------------------------

nwp_encoders_dict:
  ukv:
    _target_: pvnet.models.multimodal.encoders.encoders3d.DefaultPVNet
    _partial_: True
    in_channels: 10
    out_features: 256
    number_of_conv3d_layers: 6
    conv3d_channels: 32
    image_size_pixels: 24

#--------------------------------------------
# Sat encoder settings
#--------------------------------------------

# Ignored as premade batches were created without satellite data
# sat_encoder:
#   _target_: pvnet.models.multimodal.encoders.encoders3d.DefaultPVNet
#   _partial_: True
#   in_channels: 11
#   out_features: 256
#   number_of_conv3d_layers: 6
#   conv3d_channels: 32
#   image_size_pixels: 24

add_image_embedding_channel: False

#--------------------------------------------
# PV encoder settings
#--------------------------------------------

pv_encoder:
  _target_: pvnet.models.multimodal.site_encoders.encoders.SingleAttentionNetwork
  _partial_: True
  num_sites: 349
  out_features: 40
  num_heads: 4
  kdim: 40
  pv_id_embed_dim: 20

#--------------------------------------------
# Tabular network settings
#--------------------------------------------

output_network:
  _target_: pvnet.models.multimodal.linear_networks.networks.ResFCNet2
  _partial_: True
  fc_hidden_features: 128
  n_res_blocks: 6
  res_block_layers: 2
  dropout_frac: 0.0

embedding_dim: 16
include_sun: True
include_gsp_yield_history: False

#--------------------------------------------
# Times
#--------------------------------------------

# Foreast and time settings
history_minutes: 60
forecast_minutes: 120

min_sat_delay_minutes: 60

sat_history_minutes: 90
pv_history_minutes: 60

# These must be set for each NWP encoder
nwp_history_minutes:
  ukv: 60
nwp_forecast_minutes:
  ukv: 120

# ----------------------------------------------
# Optimizer
# ----------------------------------------------
optimizer:
  _target_: pvnet.optimizers.EmbAdamWReduceLROnPlateau
  lr: 0.0001
  weight_decay: 0.01
  amsgrad: True
  patience: 5
  factor: 0.1
  threshold: 0.002
```

In `configs/local_trainer.yaml`:

```yaml
_target_: lightning.pytorch.trainer.trainer.Trainer

accelerator: cpu # Important if running on a system without a supported GPU
devices: auto

min_epochs: null
max_epochs: null
reload_dataloaders_every_n_epochs: 0
num_sanity_val_steps: 8
fast_dev_run: false
accumulate_grad_batches: 4
log_every_n_steps: 50
```

And finally update `defaults` in the main `./configs/config.yaml` file to use
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

Assuming you ran the `save_batches.py` script to generate some premade train and
val data batches, you can now train PVNet by running:

```
python run.py
```

## Testing

You can use `python -m pytest tests` to run tests
