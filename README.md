# PVNet 2.1
[![test-release](https://github.com/openclimatefix/PVNet/actions/workflows/test-release.yml/badge.svg)](https://github.com/openclimatefix/PVNet/actions/workflows/test-release.yml)

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
