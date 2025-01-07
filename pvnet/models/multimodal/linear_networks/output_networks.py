# output_networks.py

""" 
Output networks for dynamic multimodal fusion

Defined DynamicOutputNetwork and QuantileOutputNetwork fundamentally hanlde output requirements of architecture
"""


import torch
import torch.nn.functional as F
from torch import nn
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Union

from pvnet.models.multimodal.linear_networks.basic_blocks import AbstractLinearNetwork


class DynamicOutputNetwork(AbstractLinearNetwork):
    """ Dynamic output network definition """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        use_residual: bool = True,
        output_activation: Optional[str] = None,
        quantile_output: bool = False,
        num_forecast_steps: Optional[int] = None
    ):
        # Initialisation of dynamic output network
        super().__init__(in_features=in_features, out_features=out_features)
        
        if hidden_dims is None:
            hidden_dims = [in_features * 2, in_features]
            
        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError("hidden_dims must be positive")
            
        # Construction of network layers
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.quantile_output = quantile_output
        self.num_forecast_steps = num_forecast_steps
        self.layers = nn.ModuleList()
        prev_dim = in_features
        
        for dim in hidden_dims:

            # Linear transformation / normalisatiom
            layer_block = []
            layer_block.append(nn.Linear(prev_dim, dim))
            
            if use_layer_norm:
                layer_block.append(nn.LayerNorm(dim))
                
            # Activation and dropout
            layer_block.extend([
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            
            self.layers.append(nn.Sequential(*layer_block))
            prev_dim = dim
            
        # Output layer definition
        if quantile_output and num_forecast_steps:
            final_out_features = out_features * num_forecast_steps
        else:
            final_out_features = out_features
            
        self.output_layer = nn.Linear(prev_dim, final_out_features)
        
        # Output activation definition
        if output_activation == "softmax":
            self.output_activation = nn.Softmax(dim=-1)
        elif output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = None
            
        # Optional layer norm for residual connection
        if use_residual:
            self.residual_norm = nn.LayerNorm(out_features)
            
    def reshape_quantile_output(self, x: torch.Tensor) -> torch.Tensor:

        # Reshape output for quantile predictions
        if self.quantile_output and self.num_forecast_steps:
            return x.reshape(x.shape[0], self.num_forecast_steps, -1)
        return x
            
    def forward(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        return_intermediates: bool = False
    ) -> Union[torch.Tensor, tuple]:

        # Forward pass for dynamic output network
        # Handle dict input
        if isinstance(x, dict):
            x = torch.cat(list(x.values()), dim=-1)
            
        intermediates = []
        residual = x
        
        # Process through hidden layers
        for layer in self.layers:
            x = layer(x)
            if return_intermediates:
                intermediates.append(x)
                
        # Output transform, reshape and apply residual connection
        x = self.output_layer(x)        
        x = self.reshape_quantile_output(x)        
        if self.use_residual and x.shape == residual.shape:
            x = self.residual_norm(x + residual)
            
        # Apply output activation
        if self.output_activation:
            x = self.output_activation(x)
            
        if return_intermediates:
            return x, intermediates
        return x
        
        
class QuantileOutputNetwork(DynamicOutputNetwork):
    """ Output network for quantile regression """
    
    def __init__(
        self,
        in_features: int,
        num_quantiles: int,
        num_forecast_steps: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1
    ):
        
        # Initialisation of quantile output network
        super().__init__(
            in_features=in_features,
            out_features=num_quantiles,
            hidden_dims=hidden_dims,
            dropout=dropout,
            quantile_output=True,
            num_forecast_steps=num_forecast_steps
        )
