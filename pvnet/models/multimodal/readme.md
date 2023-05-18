## Multimodal model architecture

These models fusion models to predict GSP power output based on NWP, non-HRV satellite, GSP output history, solor coordinates, and GSP ID.

The core model is `multimodel.Model`, and its architecture is shown in the diagram below. 

![multimodal_model_diagram](https://github.com/openclimatefix/PVNet/assets/41546094/118393fa-52ec-4bfe-a0a3-268c94c25f1e)

This model uses encoders which take 4D (time, channel, x, y) inputs of NWP and satellite and encode them into 1D feature vectors. Different encoders are contained inside `encoders`. 

Different choices for the fusion model are contained inside `linear_networks`. 

### Additional model architectures

The `deep_supervision.Model` network adds additional fusion model heads which predict the GSP output from only the satellite feature vector and from only the NWP feature vector. 

The `weather_residual.Model` network trains one head using the solar coords, GSP history and GSP ID, and trains a second network to learn a residual to this output from the NWP and satellite inputs. This loosely separates the predictions into "blue sky" and weather components.

The `nwp_weighting.Model` network is a simple model which learns a linear interpolation of the downward short wave radiation flux from the NWP to predict the GSP output.