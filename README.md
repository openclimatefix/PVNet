# PVNet 2.1

[![test-release](https://github.com/openclimatefix/PVNet/actions/workflows/test-release.yml/badge.svg)](https://github.com/openclimatefix/PVNet/actions/workflows/test-release.yml)

This project is used for training PVNet and running PVnet on live data.

PVNet2 largely inherits the same architecture from [PVNet1.0](https://github.com/openclimatefix/predict_pv_yield).
The NWP and satellite data are sent through some neural network which encodes them down to 1D intermediate representations.
These are concatenated together with the GSP output history, the calculated solar coordinates (azimuth and elevation) and the GSP ID which has been put through an embedding layer.
This 1D concatenated feature vector is put through an output network which outputs predictions of the future GSP yield.
National forecasts are made by adding all the GSP forecasts together.

## Setup
```bash
git clone https://github.com/openclimatefix/PVNet.git
cd PVNet
pip install -r requirements.txt
pip install git+https://github.com/SheffieldSolar/PV_Live-API
```

## Running
```bash
python run.py
```

## Development
```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

Might need to install PVLive
```
pip install git+https://github.com/SheffieldSolar/PV_Live-API#pvlive_api
```

## Testing

You can use `pytest` to run tests

## Experiments

Notes on these experiments are [here](https://docs.google.com/document/d/1fbkfkBzp16WbnCg7RDuRDvgzInA6XQu3xh4NCjV-WDA/edit?usp=sharing).
