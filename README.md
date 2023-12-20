# PVNet 2.1

[![test-release](https://github.com/openclimatefix/PVNet/actions/workflows/test-release.yml/badge.svg)](https://github.com/openclimatefix/PVNet/actions/workflows/test-release.yml)

This project is used for training PVNet and running PVnet on live data.

PVNet2 largely inherits the same architecture from
[PVNet1.0](https://github.com/openclimatefix/predict_pv_yield). The NWP and
satellite data are sent through some neural network which encodes them down to
1D intermediate representations. These are concatenated together with the GSP
output history, the calculated solar coordinates (azimuth and elevation) and the
GSP ID which has been put through an embedding layer. This 1D concatenated
feature vector is put through an output network which outputs predictions of the
future GSP yield. National forecasts are made by adding all the GSP forecasts
together.

## Setup / Installation

```bash
git clone https://github.com/openclimatefix/PVNet.git
cd PVNet
pip install -r requirements.txt
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

You will be making local amendments to these configs

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

### Generating pre-made batches of data for training/validation of PVNet

PVNet contains a script for generating batches of data suitable for training the
PVNet models.

To run the script you will need to make some modifications to the datamodule
configuration.

1. First, create your new configuration file in
   `./configs/datamodule/configiration/local_configuration.yaml` and paste the
   sample config (shown below)
2. Duplicate the `./configs/datamodule/ocf_datapipes.yaml` to
   `./configs/datamodule/_local_ocf_datapipes.yaml` and ensure the
   `configuration` key points to your newly created configuration file in
   step 1.
3. Also in this file, update the train, val & test periods to cover the data you
   have access to.
4. To get you started with your own configuration file, see the sample config
   below. Update the data paths to the location of your local GSP, NWP and PV
   datasets:

```yaml
general:
  description: Demo config
  name: demo_datamodule_config

input_data:
  default_history_minutes: 60
  default_forecast_minutes: 120

  gsp:
    gsp_zarr_path: /path/to/gsp-data.zarr
    history_minutes: 60
    forecast_minutes: 120
    time_resolution_minutes: 30
    start_datetime: "2019-01-01T00:00:00"
    end_datetime: "2019-01-08T00:00:00"
    metadata_only: false

  nwp:
    ukv:
      nwp_zarr_path: /path/to/nwp-data.zarr
      history_minutes: 60
      forecast_minutes: 120
      time_resolution_minutes: 60
      nwp_channels: # comment out channels as appropriate
        - t # live = t2m
        - dswrf
        - dlwrf
        - hcc
        - MCC
        - lcc
        - vis
        - r # live = r2
        - prate # live ~= rprate
        - si10 # 10-metre wind speed | live = unknown
      nwp_image_size_pixels_height: 24
      nwp_image_size_pixels_width: 24
      nwp_provider: ukv

  pv:
    pv_files_groups:
      - label: pvoutput.org
        pv_filename: /path/to/pv-data/pv.netcdf
        pv_metadata_filename: /path/to/pv-data/metadata.csv
    history_minutes: 60
    forecast_minutes: 0 # PVNet assumes no future PV generation
    time_resolution_minutes: 5
    start_datetime: "2019-01-01T00:00:00"
    end_datetime: "2019-01-08T00:00:00"
    pv_image_size_meters_height: 24
    pv_image_size_meters_width: 24
    pv_ml_ids: [154,155,156,158,159,160,162,164,165,166,167,168,169,171,173,177,178,179,181,182,185,186,187,188,189,190,191,192,193,197,198,199,200,202,204,205,206,208,209,211,214,215,216,217,218,219,220,221,225,229,230,232,233,234,236,242,243,245,252,254,255,256,257,258,260,261,262,265,267,268,272,273,275,276,277,280,281,282,283,287,289,291,292,293,294,295,296,297,298,301,302,303,304,306,307,309,310,311,317,318,319,320,321,322,323,325,326,329,332,333,335,336,338,340,342,344,345,346,348,349,352,354,355,356,357,360,362,363,368,369,370,371,372,374,375,376,378,380,382,384,385,388,390,391,393,396,397,398,399,400,401,403,404,405,406,407,409,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,429,431,435,437,438,440,441,444,447,450,451,453,456,457,458,459,464,465,466,467,468,470,471,473,474,476,477,479,480,481,482,485,486,488,490,491,492,493,496,498,501,503,506,507,508,509,510,511,512,513,515,516,517,519,520,521,522,524,526,527,528,531,532,536,537,538,540,541,542,543,544,545,549,550,551,552,553,554,556,557,560,561,563,566,568,571,572,575,576,577,579,580,581,582,584,585,588,590,594,595,597,600,602,603,604,606,611,613,614,616,618,620,622,623,624,625,626,628,629,630,631,636,637,638,640,641,642,644,645,646,650,651,652,653,654,655,657,660,661,662,663,666,667,668,670,675,676,679,681,683,684,685,687,696,698,701,702,703,704,706,710,722,723,724,725,727,728,729,730,732,733,734,735,736,737,]
    n_pv_systems_per_example: 128
    get_center: false
    is_live: false

  satellite:
    satellite_zarr_path: "" # Left empty to avoid using satellite data
    history_minutes: 60
    forecast_minutes: 0
    live_delay_minutes: 30
    time_resolution_minutes: 5
    satellite_channels:
      - IR_016
      - IR_039
      - IR_087
      - IR_097
      - IR_108
      - IR_120
      - IR_134
      - VIS006
      - VIS008
      - WV_062
      - WV_073
    satellite_image_size_pixels_height: 24
    satellite_image_size_pixels_width: 24
```

With your configuration in place, you can proceed to create batches. PVNet uses
[hydra](https://hydra.cc/) which enables us to pass variables via the command
line that will override the configuration defined in the `./configs` directory.

Run the save_batches.py script to create batches with the following arguments as
a minimum:

```
python scripts/save_batches.py datamodule=local_ocf_datapipes +batch_output_dir="./output" +num_train_batches=10 +num_val_batches=5
```

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

## Experiments

Notes on these experiments are
[here](https://docs.google.com/document/d/1fbkfkBzp16WbnCg7RDuRDvgzInA6XQu3xh4NCjV-WDA/edit?usp=sharing).
