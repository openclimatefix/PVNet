"""Model architecture where preidction is split into climateology and weather residual"""
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn.functional as F
from ocf_datapipes.batch import BatchKey
from torch import nn

import pvnet
from pvnet.models.base_model import BaseModel
from pvnet.models.multimodal.basic_blocks import CompleteDropoutNd, ImageEmbedding
from pvnet.models.multimodal.encoders.basic_blocks import AbstractNWPSatelliteEncoder
from pvnet.models.multimodal.encoders.encoders3d import DefaultPVNet
from pvnet.models.multimodal.linear_networks.basic_blocks import AbstractLinearNetwork
from pvnet.models.multimodal.linear_networks.networks import DefaultFCNet
from pvnet.optimizers import AbstractOptimizer


class Model(BaseModel):
    """Multi-head weather residual model.

    This architecture, which is similar to both the `multimodal.Model` and
    `deep_supervision.Model` is designed to force the network to use the information in the NWP and
    satellite data to learn the residual effect due to weather.

    Architecture is roughly as follows:

    - The GSP history*, GSP ID embedding*, and sun paramters* are concatenated into a 1D feature
        vector and passed through a neural network to produce a forecast.
    - Satellite data, if included, is put through an encoder which transforms it from 4D, with time,
        channel, height, and width dimensions to become a 1D feature vector.
    - NWP, if included, is put through a similar encoder.
    - The satellite data*, and NWP data*, are concatenated into a 1D feature vector and passed
        through another neural network to combine them and produce residual to the forecast based
        on the other data sources.
    - The residual is added to the output of the first network to produce the forecast.

    * if included

    During training we otpimise the average loss of the non-weather (i.e. not including NWP and
    satellite data) network and the weather residual network. This means the non-weather network
    should itself produce a good forecast and the weather network is forced to learn a residual.
    """

    name = "conv3d_sat_nwp_weather_residual"

    def __init__(
        self,
        image_encoder: AbstractNWPSatelliteEncoder = DefaultPVNet,
        encoder_out_features: int = 128,
        encoder_kwargs: dict = dict(),
        sat_encoder: Optional[AbstractNWPSatelliteEncoder] = None,
        sat_encoder_kwargs: Optional[dict] = None,
        output_network: AbstractLinearNetwork = DefaultFCNet,
        output_network_kwargs: dict = dict(),
        include_sat: bool = True,
        include_nwp: bool = True,
        add_image_embedding_channel: bool = False,
        include_gsp_yield_history: bool = True,
        include_sun: bool = True,
        embedding_dim: Optional[int] = 16,
        forecast_minutes: int = 30,
        history_minutes: int = 60,
        sat_history_minutes: Optional[int] = None,
        min_sat_delay_minutes: Optional[int] = 30,
        nwp_forecast_minutes: Optional[int] = None,
        nwp_history_minutes: Optional[int] = None,
        sat_image_size_pixels: int = 64,
        nwp_image_size_pixels: int = 64,
        number_sat_channels: int = 12,
        number_nwp_channels: int = 10,
        version=1,
        source_dropout=0.0,
        optimizer: AbstractOptimizer = pvnet.optimizers.Adam(),
    ):
        """Multi-head weather residual model.

        Args:
            image_encoder: Pytorch Module class used to encode the NWP data (and satellite data
                unless sat_encoder is set) from 4D into an 1D feature vector.
            encoder_out_features: Number of features of the 1D vector created by the
                `encoder_out_features` class.
            encoder_kwargs: Dictionary of optional kwargs for the `image_encoder` module.
            sat_encoder: Pytorch Module class used to encode the satellite data from 4D
                into an 1D feature vector. If not set `image_encoder` is used.
            sat_encoder_kwargs: Dictionary of optional kwargs for the `sat_encoder` module.
            output_network: Pytorch Module class used to combine the 1D features to produce the
                forecast. Also used for the ancillary networks.
            output_network_kwargs: Dictionary of optional kwargs for the `output_network` module.
            include_sat: Include satellite data.
            include_nwp: Include NWP data.
            add_image_embedding_channel: Add a channel to the NWP and satellite data with the
                embedding of the GSP ID.
            include_gsp_yield_history: Include GSP yield data.
            include_sun: Include sun azimuth and altitude data.
            embedding_dim: Number of embedding dimensions to use for GSP ID. Not included if set to
                `None`.
            forecast_minutes: The amount of minutes that should be forecasted.
            history_minutes: The default amount of historical minutes that are used.
            sat_history_minutes: Period of historical data to use for satellite data. Defaults to
                `history_minutes` if not provided.
            min_sat_delay_minutes: Minimum delay with respect to t0 of the first available satellite
                image.
            nwp_forecast_minutes: Period of future NWP forecast data to use. Defaults to
                `forecast_minutes` if not provided.
            nwp_history_minutes: Period of historical data to use for NWP data. Defaults to
                `history_minutes` if not provided.
            sat_image_size_pixels: Image size (assumed square) of the satellite data.
            nwp_image_size_pixels: Image size (assumed square) of the NWP data.
            number_sat_channels: Number of satellite channels used.
            number_nwp_channels: Number of NWP channels used.

            version: If `version=1` then the output of the non-weather forecast is fed as a feature
                into the weather residual model. If `version=0` it is not.

            source_dropout: Fraction of samples where each data source will be completely dropped
                out.

            optimizer: Optimizer factory function used for network.
        """
        self.include_gsp_yield_history = include_gsp_yield_history
        self.include_sat = include_sat
        self.include_nwp = include_nwp
        self.include_sun = include_sun
        self.embedding_dim = embedding_dim
        self.add_image_embedding_channel = add_image_embedding_channel
        self.version = version

        super().__init__(history_minutes, forecast_minutes, optimizer)

        if not (include_sat or include_nwp):
            raise ValueError("At least one of `include_sat` or `include_nwp` must be `True`.")
        assert version in [0, 1], "Version must be 0 or 1. See class docs for description."

        if include_sat:
            # We limit the history to have a delay of 15 mins in satellite data
            if sat_encoder is None:
                sat_encoder = image_encoder
                sat_encoder_kwargs = encoder_kwargs

            if sat_history_minutes is None:
                sat_history_minutes = history_minutes
            self.sat_sequence_len = (sat_history_minutes - min_sat_delay_minutes) // 5 + 1

            self.sat_encoder = sat_encoder(
                sequence_length=self.sat_sequence_len,
                image_size_pixels=sat_image_size_pixels,
                in_channels=number_sat_channels + add_image_embedding_channel,
                out_features=encoder_out_features,
                **sat_encoder_kwargs,
            )
            if add_image_embedding_channel:
                self.sat_embed = ImageEmbedding(318, self.sat_sequence_len, sat_image_size_pixels)

        if include_nwp:
            if nwp_history_minutes is None:
                nwp_history_minutes = history_minutes
            if nwp_forecast_minutes is None:
                nwp_forecast_minutes = forecast_minutes
            nwp_sequence_len = nwp_history_minutes // 60 + nwp_forecast_minutes // 60 + 1

            self.nwp_encoder = image_encoder(
                sequence_length=nwp_sequence_len,
                image_size_pixels=nwp_image_size_pixels,
                in_channels=number_nwp_channels + add_image_embedding_channel,
                out_features=encoder_out_features,
                **encoder_kwargs,
            )
            if add_image_embedding_channel:
                self.nwp_embed = ImageEmbedding(318, nwp_sequence_len, nwp_image_size_pixels)

        if self.embedding_dim:
            self.embed = nn.Embedding(num_embeddings=318, embedding_dim=self.embedding_dim)

        if self.include_sun:
            # the minus 12 is bit of hard coded smudge for pvnet
            self.sun_fc1 = nn.Linear(
                in_features=2 * (self.forecast_len + self.history_len + 1),
                out_features=16,
            )

        weather_cat_features = 0

        if include_sat:
            weather_cat_features += encoder_out_features
        if include_nwp:
            weather_cat_features += encoder_out_features
        if version == 1:
            weather_cat_features += self.forecast_len

        nonweather_cat_features = 0
        if include_gsp_yield_history:
            nonweather_cat_features += self.history_len
        if embedding_dim:
            nonweather_cat_features += embedding_dim
        if include_sun:
            nonweather_cat_features += 16

        self.simple_output_network = output_network(
            in_features=nonweather_cat_features,
            out_features=self.forecast_len,
            **output_network_kwargs,
        )

        self.weather_residual_network = nn.Sequential(
            output_network(
                in_features=weather_cat_features,
                out_features=self.forecast_len,
                **output_network_kwargs,
            ),
            # All output network return LeakyReLU activated outputs
            # However, the residual could be positive or negative
            nn.Linear(self.forecast_len, self.forecast_len),
        )

        self.source_dropout_0d = CompleteDropoutNd(0, p=source_dropout)
        self.source_dropout_3d = CompleteDropoutNd(3, p=source_dropout)

        self.save_hyperparameters()

    def encode(self, x):
        """Encode the inputs before fusion"""
        modes = OrderedDict()
        # ******************* Satellite imagery *************************
        if self.include_sat:
            # Shape: batch_size, seq_length, channel, height, width
            sat_data = x[BatchKey.satellite_actual]
            sat_data = torch.swapaxes(sat_data, 1, 2).float()  # switch time and channels
            if self.add_image_embedding_channel:
                id = x[BatchKey.gsp_id][:, 0].int()
                sat_data = self.sat_embed(sat_data, id)
            sat_data = self.source_dropout_3d(sat_data)
            modes["sat"] = self.sat_encoder(sat_data)

        # *********************** NWP Data ************************************
        if self.include_nwp:
            # shape: batch_size, seq_len, n_chans, height, width
            nwp_data = x[BatchKey.nwp].float()
            nwp_data = torch.swapaxes(nwp_data, 1, 2)  # switch time and channels
            if self.add_image_embedding_channel:
                id = x[BatchKey.gsp_id][:, 0].int()
                nwp_data = self.nwp_embed(nwp_data, id)
            nwp_data = self.source_dropout_3d(nwp_data)
            modes["nwp"] = self.nwp_encoder(nwp_data)

        # *********************** GSP Data ************************************
        # add gsp yield history
        if self.include_gsp_yield_history:
            gsp_history = x[BatchKey.gsp][:, : self.history_len].float()
            gsp_history = gsp_history.reshape(gsp_history.shape[0], -1)
            gsp_history = self.source_dropout_0d(gsp_history)
            modes["gsp"] = gsp_history

        # ********************** Embedding of GSP ID ********************
        if self.embedding_dim:
            id = x[BatchKey.gsp_id][:, 0].int()
            id_embedding = self.embed(id)
            modes["id"] = id_embedding

        if self.include_sun:
            sun = torch.cat(
                (x[BatchKey.gsp_solar_azimuth], x[BatchKey.gsp_solar_elevation]), dim=1
            ).float()
            sun = self.source_dropout_0d(sun)
            sun = self.sun_fc1(sun)
            modes["sun"] = sun
        return modes

    def forward(self, x):
        """Run normal model forward"""
        return self.multi_mode_forward(x)["weather_out"]

    def base_and_resid_forward(self, x):
        """Run forward with to predict climateology result and weather residual"""
        modes = self.encode(x)

        simple_in = OrderedDict((k, v) for k, v in modes.items() if k not in ["sat", "nwp"])
        simple_output = self.simple_output_network(simple_in)

        weather_in = OrderedDict((k, v) for k, v in modes.items() if k in ["sat", "nwp"])
        if self.version == 1:
            weather_in["y"] = simple_output
        weather_resid = self.weather_residual_network(weather_in)

        return simple_output, weather_resid

    def multi_mode_forward(self, x):
        """Predict output with multiple model heads"""
        simple_output, weather_resid = self.base_and_resid_forward(x)
        weather_out = F.leaky_relu(simple_output + weather_resid, negative_slope=0.01)
        outs = OrderedDict(simple_out=simple_output, weather_out=weather_out)
        return outs

    def training_step(self, batch, batch_idx):
        """Run training step"""
        y_hats = self.multi_mode_forward(batch)
        y = batch[BatchKey.gsp][:, -self.forecast_len :, 0]

        losses = self._calculate_common_losses(y, y_hats["weather_out"])
        losses = {f"{k}/train": v for k, v in losses.items()}

        simple_loss = F.l1_loss(y_hats["simple_out"], y)
        weather_loss = F.l1_loss(y_hats["weather_out"], y)

        # Log the loss of the network without explicit weather inputs
        losses["MAE/train/simple_loss"] = simple_loss

        loss = (weather_loss + simple_loss) / 2

        # Log the loss we actually do gradient decent on
        losses["MAE/train/multi-mode"] = loss

        self._training_accumulate_log(batch, batch_idx, losses, y_hats["weather_out"])

        return loss
