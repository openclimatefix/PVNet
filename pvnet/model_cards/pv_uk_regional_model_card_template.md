---
{{ card_data }}
---






# PVNet2

## Model Description

<!-- Provide a longer summary of what this model is/does. -->
This model class uses satellite data, numerical weather predictions, and recent Grid Service Point( GSP) PV power output to forecast the near-term (~8 hours) PV power output at all GSPs. More information can be found in the model repo [1] and experimental notes in [this google doc](https://docs.google.com/document/d/1fbkfkBzp16WbnCg7RDuRDvgzInA6XQu3xh4NCjV-WDA/edit?usp=sharing).

- **Developed by:** openclimatefix
- **Model type:** Fusion model
- **Language(s) (NLP):** en
- **License:** mit


# Training Details

## Data

<!-- This should link to a Data Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

The model is trained on data from 2019-2022 and validated on data from 2022-2023. See experimental notes in the [the google doc](https://docs.google.com/document/d/1fbkfkBzp16WbnCg7RDuRDvgzInA6XQu3xh4NCjV-WDA/edit?usp=sharing) for more details.


### Preprocessing

Data is prepared with the `ocf_data_sampler/torch_datasets/datasets/pvnet_uk` Dataset [2].


## Results

The training logs for the current model can be found here:
{{ wandb_links }}

The training logs for all model runs of PVNet2 can be found [here](https://wandb.ai/openclimatefix/pvnet2.1).

Some experimental notes can be found at in [the google doc](https://docs.google.com/document/d/1fbkfkBzp16WbnCg7RDuRDvgzInA6XQu3xh4NCjV-WDA/edit?usp=sharing)


### Hardware

Trained on a single NVIDIA Tesla T4

### Software

This model was trained using the following Open Climate Fix packages:

- [1] https://github.com/openclimatefix/PVNet
- [2] https://github.com/openclimatefix/ocf-data-sampler

The versions of these packages can be found below:
{{ package_versions }}
