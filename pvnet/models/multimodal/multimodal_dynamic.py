# multimodal_dynamic.py

from collections import OrderedDict
from typing import Optional, Dict, List, Tuple, Any, Union
import logging

import torch
from torch import nn
from ocf_datapipes.batch import BatchKey, NWPBatchKey
from omegaconf import DictConfig

import pvnet
from pvnet.models.multimodal.basic_blocks import ImageEmbedding
from pvnet.models.multimodal.encoders.dynamic_encoder import DynamicFusionEncoder
from pvnet.models.multimodal.linear_networks.basic_blocks import AbstractLinearNetwork
from pvnet.models.multimodal.site_encoders.basic_blocks import AbstractPVSitesEncoder
from pvnet.models.multimodal.multimodal_base import MultimodalBaseModel
from pvnet.optimizers import AbstractOptimizer

logger = logging.getLogger(__name__)

class Model(MultimodalBaseModel):
    name = "dynamic_fusion"

    def __init__(
        self,
        output_network: AbstractLinearNetwork,
        output_quantiles: Optional[List[float]] = None,
        nwp_encoders_dict: Optional[Dict] = None,
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
        fusion_method: str = "weighted_sum",
        forecast_minutes: int = 30,
        history_minutes: int = 60,
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
        sensor_interval_minutes: int = 30,
        wind_interval_minutes: int = 15,
        num_embeddings: Optional[int] = 318,
        timestep_intervals_to_plot: Optional[List[int]] = None,
        adapt_batches: Optional[bool] = False,
        use_weighted_loss: Optional[bool] = False,
        forecast_minutes_ignore: Optional[int] = 0,
    ):
        nn.Module.__init__(self)
        
        self.include_nwp = nwp_encoders_dict is not None and len(nwp_encoders_dict) != 0
        self.include_pv = pv_encoder is not None
        self.include_sun = include_sun
        self.include_time = include_time
        self.include_wind = wind_encoder is not None
        self.include_sensor = sensor_encoder is not None
        self.include_gsp_yield_history = include_gsp_yield_history

        if self.include_nwp:
            self.nwp_encoders_dict = nwp_encoders_dict
        if self.include_pv:
            self.pv_encoder = pv_encoder
        if self.include_wind:
            self.wind_encoder = wind_encoder
        if self.include_sensor:
            self.sensor_encoder = sensor_encoder

        self._validate_inputs(
            fusion_hidden_dim=fusion_hidden_dim,
            num_fusion_heads=num_fusion_heads,
            fusion_method=fusion_method,
            nwp_encoders_dict=nwp_encoders_dict,
            nwp_forecast_minutes=nwp_forecast_minutes,
            nwp_history_minutes=nwp_history_minutes,
            pv_encoder=pv_encoder,
            pv_history_minutes=pv_history_minutes
        )
        
        self._num_output_features = 1
        if output_quantiles:
            self._num_output_features = len(output_quantiles)       

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

        self._initialize_model_config(
            include_gsp_yield_history=include_gsp_yield_history,
            nwp_encoders_dict=nwp_encoders_dict,
            pv_encoder=pv_encoder,
            include_sun=include_sun,
            include_time=include_time,
            wind_encoder=wind_encoder,
            sensor_encoder=sensor_encoder,
            embedding_dim=embedding_dim,
            add_image_embedding_channel=add_image_embedding_channel,
            interval_minutes=interval_minutes,
            adapt_batches=adapt_batches,
            fusion_hidden_dim=fusion_hidden_dim
        )

        modality_channels = self._setup_modality_channels(
            num_embeddings=num_embeddings,
            nwp_interval_minutes=nwp_interval_minutes,
            nwp_forecast_minutes=nwp_forecast_minutes,
            nwp_history_minutes=nwp_history_minutes
        )

        self.encoder = self._initialize_fusion_encoder(
            modality_channels=modality_channels,
            fusion_hidden_dim=fusion_hidden_dim,
            num_fusion_heads=num_fusion_heads,
            fusion_dropout=fusion_dropout,
            fusion_method=fusion_method
        )

        self.output_network = output_network(
            in_features=fusion_hidden_dim,
            out_features=self.num_output_features,
        )

        self.save_hyperparameters()
        logger.info(f"Initialized {self.name} model with {len(modality_channels)} modalities")

    def _validate_inputs(self, **kwargs):
        if kwargs['fusion_hidden_dim'] <= 0:
            raise ValueError("fusion_hidden_dim must be positive")
        if kwargs['num_fusion_heads'] <= 0:
            raise ValueError("num_fusion_heads must be positive")
        if kwargs['fusion_method'] not in ["weighted_sum", "concat"]:
            raise ValueError(f"Invalid fusion method: {kwargs['fusion_method']}")
            
        if kwargs['nwp_encoders_dict']:
            if kwargs['nwp_forecast_minutes'] is None:
                raise ValueError("nwp_forecast_minutes required when using NWP encoders")
            if kwargs['nwp_history_minutes'] is None:
                raise ValueError("nwp_history_minutes required when using NWP encoders")
                
        if kwargs['pv_encoder'] is not None and kwargs['pv_history_minutes'] is None:
            raise ValueError("pv_history_minutes required when using PV encoder")

    def _initialize_model_config(self, **kwargs):
        config_params = {
            k: v for k, v in kwargs.items() 
            if not k.startswith('include_')
        }
        
        for key, value in config_params.items():
            setattr(self, key, value)
        
        if isinstance(kwargs.get('nwp_encoders_dict'), dict):
            self.nwp_encoders_dict = kwargs['nwp_encoders_dict']
        else:
            self.nwp_encoders_dict = {}

    def _setup_modality_channels(self, **kwargs) -> Dict[str, int]:
        modality_channels = {}
        
        if self.embedding_dim:
            modality_channels["embedding"] = self.embedding_dim
            
        if self.include_nwp:
            self._setup_nwp_channels(modality_channels, **kwargs)
            
        self._add_additional_channels(modality_channels)
        
        return modality_channels

    def _setup_nwp_channels(self, modality_channels: Dict[str, int], **kwargs):
        nwp_interval_minutes = kwargs.get('nwp_interval_minutes')
        if nwp_interval_minutes is None:
            nwp_interval_minutes = dict.fromkeys(self.nwp_encoders_dict.keys(), 60)

        for nwp_source, encoder in self.nwp_encoders_dict.items():
            nwp_sequence_len = (
                kwargs['nwp_history_minutes'][nwp_source] // nwp_interval_minutes[nwp_source]
                + kwargs['nwp_forecast_minutes'][nwp_source] // nwp_interval_minutes[nwp_source]
                + 1
            )
            nwp_channels = encoder.keywords["in_channels"]
            if self.add_image_embedding_channel:
                nwp_channels += 1
                self.nwp_embed_dict[nwp_source] = ImageEmbedding(
                    kwargs['num_embeddings'],
                    nwp_sequence_len,
                    encoder.image_size_pixels,
                )
            modality_channels[f"nwp/{nwp_source}"] = nwp_channels

    def _add_additional_channels(self, modality_channels: Dict[str, int]):
        if self.include_pv:
            modality_channels["pv"] = self.pv_encoder.keywords.get("num_sites", 1)
        if self.include_wind:
            modality_channels["wind"] = self.wind_encoder.keywords.get("num_sites", 1)
        if self.include_sensor:
            modality_channels["sensor"] = self.sensor_encoder.keywords.get("num_sites", 1)
        if self.include_sun:
            modality_channels["sun"] = self.fusion_hidden_dim
        if self.include_time:
            modality_channels["time"] = self.fusion_hidden_dim
        if self.include_gsp_yield_history:
            modality_channels["gsp"] = self.history_len

    def _initialize_fusion_encoder(self, modality_channels: Dict[str, int], fusion_hidden_dim: int,
                                 num_fusion_heads: int, fusion_dropout: float, fusion_method: str) -> DynamicFusionEncoder:
        modality_encoders = {}
        
        if self.include_nwp:
            for nwp_source, encoder in self.nwp_encoders_dict.items():
                modality_encoders[f"nwp/{nwp_source}"] = {
                    "image_size_pixels": encoder.image_size_pixels,
                }
        
        if self.include_pv and self.pv_encoder:
            modality_encoders["pv"] = {
                "num_sites": self.pv_encoder.keywords.get("num_sites", 1)
            }

        return DynamicFusionEncoder(
            sequence_length=self.history_len,
            image_size_pixels=224,
            modality_channels=modality_channels,
            out_features=self.num_output_features,
            modality_encoders=modality_encoders,
            cross_attention={'num_heads': num_fusion_heads, 'dropout': fusion_dropout},
            modality_gating={'dropout': fusion_dropout},
            dynamic_fusion={'fusion_method': fusion_method, 'use_residual': True},
            hidden_dim=fusion_hidden_dim,
            num_heads=num_fusion_heads,
            dropout=fusion_dropout
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.adapt_batches:
            x = self._adapt_batch(x)

        inputs = {}

        if self.include_nwp:
            self._process_nwp_data(x, inputs)

        if self.include_pv:
            inputs["pv"] = x[BatchKey.pv][:, :self.history_len + 1]

        if self.include_gsp_yield_history:
            inputs["gsp"] = x[BatchKey.gsp][:, :self.history_len].float()

        if self.embedding_dim:
            id = x[BatchKey[f"{self._target_key_name}_id"]][:, 0].int()
            inputs["embedding"] = self.embed(id)

        if self.include_wind:
            inputs["wind"] = x[BatchKey.wind][:, :self.history_len + 1]

        if self.include_sensor:
            inputs["sensor"] = x[BatchKey.sensor][:, :self.history_len + 1]

        if self.include_sun:
            sun_features = self._prepare_sun_features(x)
            inputs["sun"] = self.sun_fc1(sun_features)

        if self.include_time:
            time_features = self._prepare_time_features(x)
            inputs["time"] = self.time_fc1(time_features)

        encoded_features = self.encoder(inputs)
        output = self.output_network(encoded_features)

        if self.use_quantile_regression:
            output = output.reshape(output.shape[0], self.forecast_len, len(self.output_quantiles))

        return output, encoded_features

    def _process_nwp_data(self, x: Dict[str, torch.Tensor], inputs: Dict[str, torch.Tensor]):
        for nwp_source, nwp_encoder in self.nwp_encoders_dict.items():
            nwp_data = x[BatchKey.nwp][nwp_source][NWPBatchKey.nwp].float()
            nwp_data = torch.swapaxes(nwp_data, 1, 2)
            nwp_data = torch.clip(nwp_data, min=-50, max=50)

            if self.add_image_embedding_channel:
                id = x[BatchKey[f"{self._target_key_name}_id"]][:, 0].int()
                nwp_data = self.nwp_embed_dict[nwp_source](nwp_data, id)
                
            inputs[f"nwp/{nwp_source}"] = nwp_data

    def _adapt_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        adapted_batch = {}
        for key, value in batch.items():
            if isinstance(value, (torch.Tensor, dict)):
                adapted_batch[key] = value
            else:
                try:
                    adapted_batch[key] = torch.tensor(value)
                except:
                    adapted_batch[key] = value
        return adapted_batch

    def _preprocess_features(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        if modality == "nwp":
            return torch.clip(torch.swapaxes(x, 1, 2), min=-50, max=50)
        return x.float()

    def _prepare_time_features(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat((
            x[f"{self._target_key_name}_date_sin"],
            x[f"{self._target_key_name}_date_cos"],
            x[f"{self._target_key_name}_time_sin"],
            x[f"{self._target_key_name}_time_cos"],
        ), dim=1).float()

    def _prepare_sun_features(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat((
            x[BatchKey[f"{self._target_key_name}_solar_azimuth"]],
            x[BatchKey[f"{self._target_key_name}_solar_elevation"]],
        ), dim=1).float()

    def configure_optimizers(self):
        return self.optimizer.configure_optimizers(self.parameters())

    @property
    def num_output_features(self) -> int:
        return self._num_output_features

    @num_output_features.setter 
    def num_output_features(self, value: int):
        self._num_output_features = value
