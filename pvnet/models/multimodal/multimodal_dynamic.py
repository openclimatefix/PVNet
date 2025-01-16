# multimodal_dynamic.py

"""
Dynamic fusion model definition
"""

from collections import OrderedDict
from typing import Optional, Dict

import torch
from torch import nn
from ocf_datapipes.batch import BatchKey, NWPBatchKey
from omegaconf import DictConfig

import pvnet
from pvnet.models.multimodal.basic_blocks import ImageEmbedding
from pvnet.models.multimodal.encoders.basic_blocks import AbstractNWPSatelliteEncoder
from pvnet.models.multimodal.linear_networks.basic_blocks import AbstractLinearNetwork
from pvnet.models.multimodal.site_encoders.basic_blocks import AbstractPVSitesEncoder
from pvnet.models.multimodal.multimodal_base import MultimodalBaseModel
from pvnet.optimizers import AbstractOptimizer
from pvnet.models.multimodal.fusion_blocks import DynamicFusionModule
from pvnet.models.multimodal.attention_blocks import CrossModalAttention


class Model(MultimodalBaseModel):
    """
    Architecture summarised as follows:

    - Each modality encoded separately
    - Cross modal attention - early feature interaction
    - Dynamic weighting - modality importance
    - Weighted combination - final fused representation
    """

    name = "dynamic_fusion"

    def __init__(
        self,
        output_network: AbstractLinearNetwork,
        output_quantiles: Optional[list[float]] = None,
        nwp_encoders_dict: Optional[dict[AbstractNWPSatelliteEncoder]] = None,
        sat_encoder: Optional[AbstractNWPSatelliteEncoder] = None,
        pv_encoder: Optional[AbstractPVSitesEncoder] = None,
        wind_encoder: Optional[AbstractPVSitesEncoder] = None,
        sensor_encoder: Optional[AbstractPVSitesEncoder] = None,
        add_image_embedding_channel: bool = False,
        include_gsp_yield_history: bool = True,
        include_sun: bool = True,
        include_time: bool = False,
        embedding_dim: Optional[int] = 16,
        fusion_hidden_dim: int = 256,
        num_fusion_heads: int = 8,
        fusion_dropout: float = 0.1,
        use_cross_attention: bool = True,
        fusion_method: str = "weighted_sum",
        forecast_minutes: int = 30,
        history_minutes: int = 60,
        sat_history_minutes: Optional[int] = None,
        min_sat_delay_minutes: Optional[int] = 30,
        nwp_forecast_minutes: Optional[DictConfig] = None,
        nwp_history_minutes: Optional[DictConfig] = None,
        pv_history_minutes: Optional[int] = None,
        wind_history_minutes: Optional[int] = None,
        sensor_history_minutes: Optional[int] = None,
        sensor_forecast_minutes: Optional[int] = None,
        optimizer: AbstractOptimizer = pvnet.optimizers.Adam(),
        target_key: str = "gsp",
        interval_minutes: int = 30,
        nwp_interval_minutes: Optional[DictConfig] = None,
        pv_interval_minutes: int = 5,
        sat_interval_minutes: int = 5,
        sensor_interval_minutes: int = 30,
        wind_interval_minutes: int = 15,
        num_embeddings: Optional[int] = 318,
        timestep_intervals_to_plot: Optional[list[int]] = None,
        adapt_batches: Optional[bool] = False,
        use_weighted_loss: Optional[bool] = False,
        forecast_minutes_ignore: Optional[int] = 0,
    ):
        
        self.include_gsp_yield_history = include_gsp_yield_history
        self.include_sat = sat_encoder is not None
        self.include_nwp = nwp_encoders_dict is not None and len(nwp_encoders_dict) != 0
        self.include_pv = pv_encoder is not None
        self.include_sun = include_sun
        self.include_time = include_time
        self.include_wind = wind_encoder is not None
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
            use_weighted_loss=use_weighted_loss,
            forecast_minutes_ignore=forecast_minutes_ignore,
        )

        feature_dims = {}

        if self.include_sat:
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
                
            feature_dims["sat"] = self.sat_encoder.out_features

        if self.include_nwp:
            assert nwp_forecast_minutes is not None
            assert nwp_history_minutes is not None
            
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
                    
                feature_dims[f"nwp/{nwp_source}"] = self.nwp_encoders_dict[nwp_source].out_features

        if self.include_pv:
            assert pv_history_minutes is not None
            
            self.pv_encoder = pv_encoder(
                sequence_length=pv_history_minutes // pv_interval_minutes + 1,
                target_key_to_use=self._target_key_name,
                input_key_to_use="pv",
            )
            
            feature_dims["pv"] = self.pv_encoder.out_features

        if self.include_wind:
            if wind_history_minutes is None:
                wind_history_minutes = history_minutes

            self.wind_encoder = wind_encoder(
                sequence_length=wind_history_minutes // wind_interval_minutes + 1,
                target_key_to_use=self._target_key_name,
                input_key_to_use="wind",
            )
            
            feature_dims["wind"] = self.wind_encoder.out_features

        if self.include_sensor:
            if sensor_history_minutes is None:
                sensor_history_minutes = history_minutes
            if sensor_forecast_minutes is None:
                sensor_forecast_minutes = forecast_minutes

            self.sensor_encoder = sensor_encoder(
                sequence_length=sensor_history_minutes // sensor_interval_minutes
                + sensor_forecast_minutes // sensor_interval_minutes
                + 1,
                target_key_to_use=self._target_key_name,
                input_key_to_use="sensor",
            )
            
            feature_dims["sensor"] = self.sensor_encoder.out_features

        if self.embedding_dim:
            self.embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
            feature_dims["embedding"] = embedding_dim

        if self.include_sun:
            self.sun_fc1 = nn.Linear(
                in_features=2 * (self.forecast_len + self.forecast_len_ignore + self.history_len + 1),
                out_features=16,
            )
            feature_dims["sun"] = 16

        if self.include_time:
            self.time_fc1 = nn.Linear(
                in_features=4 * (self.forecast_len + self.forecast_len_ignore + self.history_len + 1),
                out_features=32,
            )
            feature_dims["time"] = 32

        if include_gsp_yield_history:
            feature_dims["gsp"] = self.history_len

        self.fusion_module = DynamicFusionModule(
            feature_dims=feature_dims,
            hidden_dim=fusion_hidden_dim,
            num_heads=num_fusion_heads,
            dropout=fusion_dropout,
            fusion_method=fusion_method,
            use_residual=True
        )

        if use_cross_attention:
            self.cross_attention = CrossModalAttention(
                embed_dim=fusion_hidden_dim,
                num_heads=num_fusion_heads,
                dropout=fusion_dropout,
                num_modalities=len(feature_dims)
            )
        else:
            self.cross_attention = None

        self.output_network = output_network(
            in_features=fusion_hidden_dim,
            out_features=self.num_output_features,
        )

        self.save_hyperparameters()

    def forward(self, x):

        if self.adapt_batches:
            x = self._adapt_batch(x)

        encoded_features = OrderedDict()

        if self.include_sat:
            sat_data = x[BatchKey.satellite_actual][:, : self.sat_sequence_len]
            sat_data = torch.swapaxes(sat_data, 1, 2).float()

            if self.add_image_embedding_channel:
                id = x[BatchKey[f"{self._target_key_name}_id"]][:, 0].int()
                sat_data = self.sat_embed(sat_data, id)
            encoded_features["sat"] = self.sat_encoder(sat_data)

        if self.include_nwp:
            for nwp_source in self.nwp_encoders_dict:
                nwp_data = x[BatchKey.nwp][nwp_source][NWPBatchKey.nwp].float()
                nwp_data = torch.swapaxes(nwp_data, 1, 2)
                nwp_data = torch.clip(nwp_data, min=-50, max=50)

                if self.add_image_embedding_channel:
                    id = x[BatchKey[f"{self._target_key_name}_id"]][:, 0].int()
                    nwp_data = self.nwp_embed_dict[nwp_source](nwp_data, id)

                encoded_features[f"nwp/{nwp_source}"] = self.nwp_encoders_dict[nwp_source](nwp_data)

        if self.include_pv:
            if self._target_key_name != "pv":
                encoded_features["pv"] = self.pv_encoder(x)
            else:
                x_tmp = x.copy()
                x_tmp[BatchKey.pv] = x_tmp[BatchKey.pv][:, : self.history_len + 1]
                encoded_features["pv"] = self.pv_encoder(x_tmp)

        if self.include_gsp_yield_history:
            gsp_history = x[BatchKey.gsp][:, : self.history_len].float()
            encoded_features["gsp"] = gsp_history.reshape(gsp_history.shape[0], -1)

        if self.embedding_dim:
            id = x[BatchKey[f"{self._target_key_name}_id"]][:, 0].int()
            encoded_features["embedding"] = self.embed(id)

        if self.include_wind:
            if self._target_key_name != "wind":
                encoded_features["wind"] = self.wind_encoder(x)
            else:
                x_tmp = x.copy()
                x_tmp[BatchKey.wind] = x_tmp[BatchKey.wind][:, : self.history_len + 1]
                encoded_features["wind"] = self.wind_encoder(x_tmp)

        if self.include_sensor:
            if self._target_key_name != "sensor":
                encoded_features["sensor"] = self.sensor_encoder(x)
            else:
                x_tmp = x.copy()
                x_tmp[BatchKey.sensor] = x_tmp[BatchKey.sensor][:, : self.history_len + 1]
                encoded_features["sensor"] = self.sensor_encoder(x_tmp)

        if self.include_sun:
            sun = torch.cat(
                (
                    x[BatchKey[f"{self._target_key_name}_solar_azimuth"]],
                    x[BatchKey[f"{self._target_key_name}_solar_elevation"]],
                ),
                dim=1,
            ).float()
            encoded_features["sun"] = self.sun_fc1(sun)


        if self.include_time:
            time = torch.cat(
                (
                    x[f"{self._target_key_name}_date_sin"],
                    x[f"{self._target_key_name}_date_cos"],
                    x[f"{self._target_key_name}_time_sin"],
                    x[f"{self._target_key_name}_time_cos"],
                ),
                dim=1,
            ).float()
            encoded_features["time"] = self.time_fc1(time)

        if self.cross_attention is not None and len(encoded_features) > 1:
            encoded_features = self.cross_attention(encoded_features)

        fused_features = self.fusion_module(encoded_features)
        out = self.output_network(fused_features)

        if self.use_quantile_regression:
            out = out.reshape(out.shape[0], self.forecast_len, len(self.output_quantiles))

        return out
