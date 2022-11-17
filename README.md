# PVNet
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
