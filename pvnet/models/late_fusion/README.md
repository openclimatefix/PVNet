## Multimodal model architecture

These models fusion models to predict GSP power output based on NWP, non-HRV satellite, GSP output history, solor coordinates, and GSP ID.

The core model is `late_fusion.LateFusionModel`, and its architecture is shown in the diagram below.

![multimodal_model_diagram](https://github.com/openclimatefix/PVNet/assets/41546094/118393fa-52ec-4bfe-a0a3-268c94c25f1e)

This model uses encoders which take 4D (time, channel, x, y) inputs of NWP and satellite and encode them into 1D feature vectors. Different encoders are contained inside `encoders`.

Different choices for the fusion model are contained inside `linear_networks`.
