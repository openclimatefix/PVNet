"""The default composite model architecture for PVNet"""

from collections import OrderedDict
from typing import Optional

import torch
from ocf_datapipes.utils.consts import BatchKey
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
    """Neural network which combines information from different sources.

    Architecture is roughly as follows:

    - Satellite data, if included, is put through an encoder which transforms it from 4D, with time,
        channel, height, and width dimensions to become a 1D feature vector.
    - NWP, if included, is put through a similar encoder.
    - The satellite data*, NWP data*, GSP history*, GSP ID embedding*, and sun paramters* are
        concatenated into a 1D feature vector and passed through another neural network to combine
        them and produce a forecast.

    * if included
    """

    name = "conv3d_sat_nwp"

    def __init__(
        self,
        output_quantiles: Optional[list[float]] = None,
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
        source_dropout=0.0,
        optimizer: AbstractOptimizer = pvnet.optimizers.Adam(),
    ):
        """Neural network which combines information from different sources.

        Args:
            output_quantiles: A list of float (0.0, 1.0) quantiles to predict values for. If set to
                None the output is a single value.
            image_encoder: Pytorch Module class used to encode the NWP data (and satellite data
                unless sat_encoder is set) from 4D into an 1D feature vector.
            encoder_out_features: Number of features of the 1D vector created by the
                `encoder_out_features` class.
            encoder_kwargs: Dictionary of optional kwargs for the `image_encoder` module.
            sat_encoder: Pytorch Module class used to encode the satellite data from 4D
                into an 1D feature vector. If not set `image_encoder` is used.
            sat_encoder_kwargs: Dictionary of optional kwargs for the `sat_encoder` module.
            output_network: Pytorch Module class used to combine the 1D features to produce the
                forecast.
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

        super().__init__(history_minutes, forecast_minutes, optimizer, output_quantiles)

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
                in_features=2 * (self.forecast_len_30 + self.history_len_30 + 1),
                out_features=16,
            )

        fc_in_features = 0
        if include_sat:
            fc_in_features += encoder_out_features
        if include_nwp:
            fc_in_features += encoder_out_features
        if include_gsp_yield_history:
            fc_in_features += self.history_len_30
        if embedding_dim:
            fc_in_features += embedding_dim
        if include_sun:
            fc_in_features += 16

        self.output_network = output_network(
            in_features=fc_in_features,
            out_features=self.num_output_features,
            **output_network_kwargs,
        )

        self.source_dropout_0d = CompleteDropoutNd(0, p=source_dropout)
        self.source_dropout_3d = CompleteDropoutNd(3, p=source_dropout)

        self.save_hyperparameters()

    def forward(self, x):
        """Run model forward"""
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
            gsp_history = x[BatchKey.gsp][:, : self.history_len_30].float()
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

        out = self.output_network(modes)

        if self.use_quantile_regression:
            # Shape: batch_size, seq_length * num_quantiles
            out = out.reshape(out.shape[0], self.forecast_len_30, len(self.output_quantiles))

        return out
