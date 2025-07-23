---
{{ card_data }}
---






# WindNet

## Model Description

<!-- Provide a longer summary of what this model is/does. -->
This model class uses numerical weather predictions from providers such as ECMWF to forecast the wind power in North West India over the next 48 hours at 15 minute granularity. More information can be found in the model repo [1] and experimental notes [here](https://github.com/openclimatefix/PVNet/tree/main/experiments/india).


- **Developed by:** openclimatefix
- **Model type:** Fusion model
- **Language(s) (NLP):** en
- **License:** mit


# Training Details

## Data

<!-- This should link to a Data Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

The model is trained on data from 2019-2022 and validated on data from 2022-2023. See experimental notes [here](https://github.com/openclimatefix/PVNet/tree/main/experiments/india)


### Preprocessing

Data is prepared with the `ocf_data_sampler/torch_datasets/datasets/site` Dataset [2].


## Results

The training logs for the current model can be found here:
{{ wandb_links }}


### Hardware

Trained on a single NVIDIA Tesla T4

### Software

This model was trained using the following Open Climate Fix packages:

- [1] https://github.com/openclimatefix/PVNet
- [2] https://github.com/openclimatefix/ocf-data-sampler

The versions of these packages can be found below:
{{ package_versions }}
