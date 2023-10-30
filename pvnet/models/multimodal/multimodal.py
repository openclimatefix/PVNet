"""The default composite model architecture for PVNet"""

from collections import OrderedDict
from typing import Optional

import torch
from ocf_datapipes.utils.consts import BatchKey
from torch import nn

import pvnet
from pvnet.models.base_model import BaseModel
from pvnet.models.multimodal.basic_blocks import ImageEmbedding
from pvnet.models.multimodal.encoders.basic_blocks import AbstractNWPSatelliteEncoder
from pvnet.models.multimodal.linear_networks.basic_blocks import AbstractLinearNetwork
from pvnet.models.multimodal.site_encoders.basic_blocks import AbstractPVSitesEncoder
from pvnet.optimizers import AbstractOptimizer


class Model(BaseModel):
    """Neural network which combines information from different sources

    Architecture is roughly as follows:

    - Satellite data, if included, is put through an encoder which transforms it from 4D, with time,
        channel, height, and width dimensions to become a 1D feature vector.
    - NWP, if included, is put through a similar encoder.
    - PV site-level data, if included, is put through an encoder which transforms it from 2D, with
        time and system-ID dimensions, to become a 1D feature vector.
    - The satellite features*, NWP features*, PV site-level features*, GSP ID embedding*, and sun
        paramters* are concatenated into a 1D feature vector and passed through another neural
        network to combine them and produce a forecast.

    * if included
    """

    name = "conv3d_sat_nwp"

    def __init__(
        self,
        output_network: AbstractLinearNetwork,
        output_quantiles: Optional[list[float]] = None,
        nwp_encoder: Optional[AbstractNWPSatelliteEncoder] = None,
        sat_encoder: Optional[AbstractNWPSatelliteEncoder] = None,
        pv_encoder: Optional[AbstractPVSitesEncoder] = None,
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
        pv_history_minutes: Optional[int] = None,
        optimizer: AbstractOptimizer = pvnet.optimizers.Adam(),
    ):
        """Neural network which combines information from different sources.

        Notes:
            In the args, where it says a module `m` is partially instantiated, it means that a
            normal pytorch module will be returned by running `mod = m(**kwargs)`. In this library,
            this partial instantiation is generally achieved using partial instantiation via hydra.
            However, the arg is still valid as long as `m(**kwargs)` returns a valid pytorch module
            - for example if `m` is a regular function.

        Args:
            output_network: A partially instatiated pytorch Module class used to combine the 1D
                features to produce the forecast.
            output_quantiles: A list of float (0.0, 1.0) quantiles to predict values for. If set to
                None the output is a single value.
            nwp_encoder: A partially instatiated pytorch Module class used to encode the NWP data
                from 4D into an 1D feature vector.
            sat_encoder: A partially instatiated pytorch Module class used to encode the satellite
                data from 4D into an 1D feature vector.
            pv_encoder: A partially instatiated pytorch Module class used to encode the site-level
                PV data from 2D into an 1D feature vector.
            add_image_embedding_channel: Add a channel to the NWP and satellite data with the
                embedding of the GSP ID.
            include_gsp_yield_history: Include GSP yield data.
            include_sun: Include sun azimuth and altitude data.
            embedding_dim: Number of embedding dimensions to use for GSP ID. Not included if set to
                `None`.
            forecast_minutes: The amount of minutes that should be forecasted.
            history_minutes: The default amount of historical minutes that are used.
            sat_history_minutes: Length of recent observations used for satellite inputs. Defaults
                to `history_minutes` if not provided.
            min_sat_delay_minutes: Minimum delay with respect to t0 of the latest available
                satellite image.
            nwp_forecast_minutes: Period of future NWP forecast data used as input. Defaults to
                `forecast_minutes` if not provided.
            nwp_history_minutes: Period of historical NWP forecast used as input. Defaults to
                `history_minutes` if not provided.
            pv_history_minutes: Length of recent site-level PV data data used as input. Defaults to
                `history_minutes` if not provided.
            optimizer: Optimizer factory function used for network.
        """
        self.include_gsp_yield_history = include_gsp_yield_history
        self.include_sat = sat_encoder is not None
        self.include_nwp = nwp_encoder is not None
        self.include_pv = pv_encoder is not None
        self.include_sun = include_sun
        self.embedding_dim = embedding_dim
        self.add_image_embedding_channel = add_image_embedding_channel

        super().__init__(history_minutes, forecast_minutes, optimizer, output_quantiles)

        # Number of features expected by the output_network
        # Add to this as network pices are constructed
        fusion_input_features = 0

        if self.include_sat:
            # We limit the history to have a delay of 15 mins in satellite data

            if sat_history_minutes is None:
                sat_history_minutes = history_minutes

            self.sat_sequence_len = (sat_history_minutes - min_sat_delay_minutes) // 5 + 1

            self.sat_encoder = sat_encoder(
                sequence_length=self.sat_sequence_len,
                in_channels=sat_encoder.keywords["in_channels"] + add_image_embedding_channel,
            )
            if add_image_embedding_channel:
                self.sat_embed = ImageEmbedding(
                    318, self.sat_sequence_len, self.sat_encoder.image_size_pixels
                )

            # Update num features
            fusion_input_features += self.sat_encoder.out_features

        if self.include_nwp:
            if nwp_history_minutes is None:
                nwp_history_minutes = history_minutes
            if nwp_forecast_minutes is None:
                nwp_forecast_minutes = forecast_minutes
            nwp_sequence_len = nwp_history_minutes // 60 + nwp_forecast_minutes // 60 + 1

            self.nwp_encoder = nwp_encoder(
                sequence_length=nwp_sequence_len,
                in_channels=nwp_encoder.keywords["in_channels"] + add_image_embedding_channel,
            )
            if add_image_embedding_channel:
                self.nwp_embed = ImageEmbedding(
                    318, nwp_sequence_len, self.nwp_encoder.image_size_pixels
                )

            # Update num features
            fusion_input_features += self.nwp_encoder.out_features

        if self.include_pv:
            if pv_history_minutes is None:
                pv_history_minutes = history_minutes

            self.pv_encoder = pv_encoder(
                sequence_length=pv_history_minutes // 5 + 1,
            )

            # Update num features
            fusion_input_features += self.pv_encoder.out_features

        if self.embedding_dim:
            self.embed = nn.Embedding(num_embeddings=318, embedding_dim=embedding_dim)

            # Update num features
            fusion_input_features += embedding_dim

        if self.include_sun:
            # the minus 12 is bit of hard coded smudge for pvnet
            self.sun_fc1 = nn.Linear(
                in_features=2 * (self.forecast_len_30 + self.history_len_30 + 1),
                out_features=16,
            )

            # Update num features
            fusion_input_features += 16

        if include_gsp_yield_history:
            # Update num features
            fusion_input_features += self.history_len_30

        self.output_network = output_network(
            in_features=fusion_input_features,
            out_features=self.num_output_features,
        )

        self.save_hyperparameters()

    def forward(self, x):
        """Run model forward"""
        modes = OrderedDict()
        # ******************* Satellite imagery *************************
        if self.include_sat:
            # Shape: batch_size, seq_length, channel, height, width
            sat_data = x[BatchKey.satellite_actual][:, : self.sat_sequence_len]
            sat_data = torch.swapaxes(sat_data, 1, 2).float()  # switch time and channels
            if self.add_image_embedding_channel:
                id = x[BatchKey.gsp_id][:, 0].int()
                sat_data = self.sat_embed(sat_data, id)
            modes["sat"] = self.sat_encoder(sat_data)

        # *********************** NWP Data ************************************
        if self.include_nwp:
            # shape: batch_size, seq_len, n_chans, height, width
            nwp_data = x[BatchKey.nwp].float()
            nwp_data = torch.swapaxes(nwp_data, 1, 2)  # switch time and channels
            if self.add_image_embedding_channel:
                id = x[BatchKey.gsp_id][:, 0].int()
                nwp_data = self.nwp_embed(nwp_data, id)
            modes["nwp"] = self.nwp_encoder(nwp_data)

        # *********************** PV Data *************************************
        # Add site-level PV yield
        if self.include_pv:
            modes["pv"] = self.pv_encoder(x)

        # *********************** GSP Data ************************************
        # add gsp yield history
        if self.include_gsp_yield_history:
            gsp_history = x[BatchKey.gsp][:, : self.history_len_30].float()
            gsp_history = gsp_history.reshape(gsp_history.shape[0], -1)
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
            sun = self.sun_fc1(sun)
            modes["sun"] = sun

        out = self.output_network(modes)

        if self.use_quantile_regression:
            # Shape: batch_size, seq_length * num_quantiles
            out = out.reshape(out.shape[0], self.forecast_len_30, len(self.output_quantiles))

        return out
