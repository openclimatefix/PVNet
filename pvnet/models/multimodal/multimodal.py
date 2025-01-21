"""The default composite model architecture for PVNet"""

from collections import OrderedDict
from typing import Optional

import torch
from omegaconf import DictConfig
from torch import nn

import pvnet
from pvnet.models.multimodal.basic_blocks import ImageEmbedding
from pvnet.models.multimodal.encoders.basic_blocks import AbstractNWPSatelliteEncoder
from pvnet.models.multimodal.linear_networks.basic_blocks import AbstractLinearNetwork
from pvnet.models.multimodal.multimodal_base import MultimodalBaseModel
from pvnet.models.multimodal.site_encoders.basic_blocks import AbstractSitesEncoder
from pvnet.optimizers import AbstractOptimizer


class Model(MultimodalBaseModel):
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
        nwp_encoders_dict: Optional[dict[AbstractNWPSatelliteEncoder]] = None,
        sat_encoder: Optional[AbstractNWPSatelliteEncoder] = None,
        pv_encoder: Optional[AbstractSitesEncoder] = None,
        sensor_encoder: Optional[AbstractSitesEncoder] = None,
        add_image_embedding_channel: bool = False,
        include_gsp_yield_history: bool = True,
        include_sun: bool = True,
        include_time: bool = False,
        embedding_dim: Optional[int] = 16,
        forecast_minutes: int = 30,
        history_minutes: int = 60,
        sat_history_minutes: Optional[int] = None,
        min_sat_delay_minutes: Optional[int] = 30,
        nwp_forecast_minutes: Optional[DictConfig] = None,
        nwp_history_minutes: Optional[DictConfig] = None,
        pv_history_minutes: Optional[int] = None,
        sensor_history_minutes: Optional[int] = None,
        sensor_forecast_minutes: Optional[int] = None,
        optimizer: AbstractOptimizer = pvnet.optimizers.Adam(),
        target_key: str = "gsp",
        interval_minutes: int = 30,
        nwp_interval_minutes: Optional[DictConfig] = None,
        pv_interval_minutes: int = 5,
        sat_interval_minutes: int = 5,
        sensor_interval_minutes: int = 30,
        num_embeddings: Optional[int] = 318,
        timestep_intervals_to_plot: Optional[list[int]] = None,
        adapt_batches: Optional[bool] = False,
        forecast_minutes_ignore: Optional[int] = 0,
    ):
        """Neural network which combines information from different sources.

        Notes:
            In the args, where it says a module `m` is partially instantiated, it means that a
            normal pytorch module will be returned by running `mod = m(**kwargs)`. In this library,
            this partial instantiation is generally achieved using partial instantiation via hydra.
            However, the arg is still valid as long as `m(**kwargs)` returns a valid pytorch module
            - for example if `m` is a regular function.

        Args:
            output_network: A partially instantiated pytorch Module class used to combine the 1D
                features to produce the forecast.
            output_quantiles: A list of float (0.0, 1.0) quantiles to predict values for. If set to
                None the output is a single value.
            nwp_encoders_dict: A dictionary of partially instantiated pytorch Module class used to
                encode the NWP data from 4D into a 1D feature vector from different sources.
            sat_encoder: A partially instantiated pytorch Module class used to encode the satellite
                data from 4D into a 1D feature vector.
            pv_encoder: A partially instantiated pytorch Module class used to encode the site-level
                PV data from 2D into a 1D feature vector.
            add_image_embedding_channel: Add a channel to the NWP and satellite data with the
                embedding of the GSP ID.
            include_gsp_yield_history: Include GSP yield data.
            include_sun: Include sun azimuth and altitude data.
            include_time: Include sine and cosine of dates and times.
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
            pv_history_minutes: Length of recent site-level PV data used as
            input. Defaults to `history_minutes` if not provided.
            optimizer: Optimizer factory function used for network.
            target_key: The key of the target variable in the batch.
            interval_minutes: The interval between each sample of the target data
            nwp_interval_minutes: Dictionary of the intervals between each sample of the NWP
                data for each source
            pv_interval_minutes: The interval between each sample of the PV data
            sat_interval_minutes: The interval between each sample of the satellite data
            sensor_interval_minutes: The interval between each sample of the sensor data
            num_embeddings: The number of dimensions to use for the image embedding
            timestep_intervals_to_plot: Intervals, in timesteps, to plot in
            addition to the full forecast
            sensor_encoder: Encoder for sensor data
            sensor_history_minutes: Length of recent sensor data used as input.
            sensor_forecast_minutes: Length of forecast sensor data used as input.
            adapt_batches: If set to true, we attempt to slice the batches to the expected shape for
                the model to use. This allows us to overprepare batches and slice from them for the
                data we need for a model run.
            forecast_minutes_ignore: Number of forecast minutes to ignore when calculating losses.
                For example if set to 60, the model doesnt predict the first 60 minutes
        """

        self.include_gsp_yield_history = include_gsp_yield_history
        self.include_sat = sat_encoder is not None
        self.include_nwp = nwp_encoders_dict is not None and len(nwp_encoders_dict) != 0
        self.include_pv = pv_encoder is not None
        self.include_sun = include_sun
        self.include_time = include_time
        self.include_sensor = sensor_encoder is not None
        self.embedding_dim = embedding_dim
        self.add_image_embedding_channel = add_image_embedding_channel
        self.interval_minutes = interval_minutes
        self.min_sat_delay_minutes = min_sat_delay_minutes
        self.adapt_batches = adapt_batches

        super().__init__(
            history_minutes=history_minutes,
            forecast_minutes=forecast_minutes,
            optimizer=optimizer,
            output_quantiles=output_quantiles,
            target_key=target_key,
            interval_minutes=interval_minutes,
            timestep_intervals_to_plot=timestep_intervals_to_plot,
            forecast_minutes_ignore=forecast_minutes_ignore,
        )

        # Number of features expected by the output_network
        # Add to this as network pieces are constructed
        fusion_input_features = 0

        if self.include_sat:
            # Param checks
            assert sat_history_minutes is not None

            self.sat_sequence_len = (
                sat_history_minutes - min_sat_delay_minutes
            ) // sat_interval_minutes + 1

            self.sat_encoder = sat_encoder(
                sequence_length=self.sat_sequence_len,
                in_channels=sat_encoder.keywords["in_channels"] + add_image_embedding_channel,
            )
            if add_image_embedding_channel:
                self.sat_embed = ImageEmbedding(
                    num_embeddings, self.sat_sequence_len, self.sat_encoder.image_size_pixels
                )

            # Update num features
            fusion_input_features += self.sat_encoder.out_features

        if self.include_nwp:
            # Param checks
            assert nwp_forecast_minutes is not None
            assert nwp_history_minutes is not None

            # For each NWP encoder the forecast and history minutes must be set
            assert set(nwp_encoders_dict.keys()) == set(nwp_forecast_minutes.keys())
            assert set(nwp_encoders_dict.keys()) == set(nwp_history_minutes.keys())

            if nwp_interval_minutes is None:
                nwp_interval_minutes = dict.fromkeys(nwp_encoders_dict.keys(), 60)

            self.nwp_encoders_dict = torch.nn.ModuleDict()
            if add_image_embedding_channel:
                self.nwp_embed_dict = torch.nn.ModuleDict()

            for nwp_source in nwp_encoders_dict.keys():
                nwp_sequence_len = (
                    nwp_history_minutes[nwp_source] // nwp_interval_minutes[nwp_source]
                    + nwp_forecast_minutes[nwp_source] // nwp_interval_minutes[nwp_source]
                    + 1
                )

                self.nwp_encoders_dict[nwp_source] = nwp_encoders_dict[nwp_source](
                    sequence_length=nwp_sequence_len,
                    in_channels=(
                        nwp_encoders_dict[nwp_source].keywords["in_channels"]
                        + add_image_embedding_channel
                    ),
                )
                if add_image_embedding_channel:
                    self.nwp_embed_dict[nwp_source] = ImageEmbedding(
                        num_embeddings,
                        nwp_sequence_len,
                        self.nwp_encoders_dict[nwp_source].image_size_pixels,
                    )

                # Update num features
                fusion_input_features += self.nwp_encoders_dict[nwp_source].out_features

        if self.include_pv:
            assert pv_history_minutes is not None

            self.pv_encoder = pv_encoder(
                sequence_length=pv_history_minutes // pv_interval_minutes + 1,
                target_key_to_use=self._target_key,
                input_key_to_use="site",
            )

            # Update num features
            fusion_input_features += self.pv_encoder.out_features

        if self.include_sensor:
            if sensor_history_minutes is None:
                sensor_history_minutes = history_minutes
            if sensor_forecast_minutes is None:
                sensor_forecast_minutes = forecast_minutes

            self.sensor_encoder = sensor_encoder(
                sequence_length=sensor_history_minutes // sensor_interval_minutes
                + sensor_forecast_minutes // sensor_interval_minutes
                + 1,
                target_key_to_use=self._target_key,
                input_key_to_use="sensor",
            )

            # Update num features
            fusion_input_features += self.sensor_encoder.out_features

        if self.embedding_dim:
            self.embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

            # Update num features
            fusion_input_features += embedding_dim

        if self.include_sun:
            self.sun_fc1 = nn.Linear(
                in_features=2
                * (self.forecast_len + self.forecast_len_ignore + self.history_len + 1),
                out_features=16,
            )

            # Update num features
            fusion_input_features += 16

        if self.include_time:
            self.time_fc1 = nn.Linear(
                in_features=4
                * (self.forecast_len + self.forecast_len_ignore + self.history_len + 1),
                out_features=32,
            )

            # Update num features
            fusion_input_features += 32

        if include_gsp_yield_history:
            # Update num features
            fusion_input_features += self.history_len

        self.output_network = output_network(
            in_features=fusion_input_features,
            out_features=self.num_output_features,
        )

        self.save_hyperparameters()

    def forward(self, x):
        """Run model forward"""

        if self.adapt_batches:
            x = self._adapt_batch(x)

        modes = OrderedDict()
        # ******************* Satellite imagery *************************
        if self.include_sat:
            # Shape: batch_size, seq_length, channel, height, width
            sat_data = x["satellite_actual"][:, : self.sat_sequence_len]
            sat_data = torch.swapaxes(sat_data, 1, 2).float()  # switch time and channels

            if self.add_image_embedding_channel:
                id = x[f"{self._target_key}_id"].int()
                sat_data = self.sat_embed(sat_data, id)
            modes["sat"] = self.sat_encoder(sat_data)

        # *********************** NWP Data ************************************
        if self.include_nwp:
            # Loop through potentially many NMPs
            for nwp_source in self.nwp_encoders_dict:
                # shape: batch_size, seq_len, n_chans, height, width
                nwp_data = x["nwp"][nwp_source]["nwp"].float()
                nwp_data = torch.swapaxes(nwp_data, 1, 2)  # switch time and channels
                # Some NWP variables can overflow into NaNs when normalised if they have extreme
                # tails
                nwp_data = torch.clip(nwp_data, min=-50, max=50)

                if self.add_image_embedding_channel:
                    id = x[f"{self._target_key}_id"].int()
                    nwp_data = self.nwp_embed_dict[nwp_source](nwp_data, id)

                nwp_out = self.nwp_encoders_dict[nwp_source](nwp_data)
                modes[f"nwp/{nwp_source}"] = nwp_out

        # *********************** Site Data *************************************
        # Add site-level PV yield
        if self.include_pv:
            if self._target_key != "site":
                modes["site"] = self.pv_encoder(x)
            else:
                # Target is PV, so only take the history
                # Copy batch
                x_tmp = x.copy()
                x_tmp["site"] = x_tmp["site"][:, : self.history_len + 1]
                modes["site"] = self.pv_encoder(x_tmp)

        # *********************** GSP Data ************************************
        # add gsp yield history
        if self.include_gsp_yield_history:
            gsp_history = x["gsp"][:, : self.history_len].float()
            gsp_history = gsp_history.reshape(gsp_history.shape[0], -1)
            modes["gsp"] = gsp_history

        # ********************** Embedding of GSP/Site ID ********************
        if self.embedding_dim:
            id = x[f"{self._target_key}_id"].int()
            id_embedding = self.embed(id)
            modes["id"] = id_embedding

        if self.include_sun:
            sun = torch.cat(
                (
                    x[f"{self._target_key}_solar_azimuth"],
                    x[f"{self._target_key}_solar_elevation"],
                ),
                dim=1,
            ).float()
            sun = self.sun_fc1(sun)
            modes["sun"] = sun

        if self.include_time:
            time = torch.cat(
                (
                    x[f"{self._target_key}_date_sin"],
                    x[f"{self._target_key}_date_cos"],
                    x[f"{self._target_key}_time_sin"],
                    x[f"{self._target_key}_time_cos"],
                ),
                dim=1,
            ).float()
            time = self.time_fc1(time)
            modes["time"] = time

        out = self.output_network(modes)

        if self.use_quantile_regression:
            # Shape: batch_size, seq_length * num_quantiles
            out = out.reshape(out.shape[0], self.forecast_len, len(self.output_quantiles))

        return out
