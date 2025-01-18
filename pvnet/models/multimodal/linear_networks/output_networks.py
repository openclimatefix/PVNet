# output_networks.py

""" 
Output networks for dynamic multimodal fusion

Defined DynamicOutputNetwork and QuantileOutputNetwork fundamentally hanlde output requirements of architecture

These networks process fused multimodal representations to generate outputs
"""


import torch
import torch.nn.functional as F
from torch import nn
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Union

from pvnet.models.multimodal.linear_networks.basic_blocks import AbstractLinearNetwork


class DynamicOutputNetwork(AbstractLinearNetwork):
    """ Dynamic output network definition """
    
    # Input and output dimensions specified here
    # Defines feature mapping ℝ^n → ℝ^m
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
        self.out_features = out_features
        
        # Default hidden architecture
        # h_i ∈ ℝ^{d_i}, where d_i = [2n, n]
        if hidden_dims is None:
            hidden_dims = [in_features * 2, in_features]
            
        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError("hidden_dims must be positive")
            
        # Construction of network layers - config
        # Network architecture parameters θ
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.quantile_output = quantile_output
        self.num_forecast_steps = num_forecast_steps

        # Construction of hidden layers
        # H_i: ℝ^{d_i} → ℝ^{d_{i+1}}
        # Sequential transformation φ(x) = Dropout(ReLU(LayerNorm(Wx + b)))
        self.layers = nn.ModuleList()
        prev_dim = in_features
        
        for dim in hidden_dims:

            # Affine transformation followed by distribution normalisation
            layer_block = []
            layer_block.append(nn.Linear(prev_dim, dim))
            
            if use_layer_norm:
                layer_block.append(nn.LayerNorm(dim))
                
            # Non-linear activation and stochastic regularisation
            layer_block.extend([
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            
            self.layers.append(nn.Sequential(*layer_block))
            prev_dim = dim
            
        # Output layer transformation definition
        # f: ℝ^{d_L} → ℝ^m
        # Projection mapping P: ℝ^d → ℝ^{m×t} for temporal quantile predictions
        if quantile_output and num_forecast_steps:
            final_out_features = out_features * num_forecast_steps
        else:
            final_out_features = out_features
            
        self.output_layer = nn.Linear(prev_dim, final_out_features)
        
        # Output activation definition
        # ψ: ℝ^m → [0,1]^m
        if output_activation == "softmax":
            self.output_activation = nn.Softmax(dim=-1)
        elif output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = None

        # Optional layer norm and residual projection
        # g: ℝ^n → ℝ^m
        if use_residual:
            if quantile_output and num_forecast_steps:
                self.residual_norm = nn.LayerNorm(out_features)
            else:
                final_out_features = out_features * num_forecast_steps if quantile_output and num_forecast_steps else out_features
                self.residual_norm = nn.LayerNorm(final_out_features)
            self.residual_proj = nn.Linear(in_features, out_features)
            
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
        # Concatenate multimodal inputs if dict provided
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

        if self.use_residual:
            # Apply residual projection transformation
            projected_residual = self.residual_proj(residual)
            if self.quantile_output and self.num_forecast_steps:

                # Apply residual mapping followed by normalisation
                projected_residual = projected_residual.reshape(x.shape[0], x.shape[2])

                # Collapse temporal dimensions for normalisation 
                # ℝ^{B×T×F} → ℝ^{BT×F}
                x = x.reshape(-1, x.shape[2])
                x = self.residual_norm(x + projected_residual.repeat(self.num_forecast_steps, 1))

                # Restore tensor dimensionality
                # ℝ^{BT×F} → ℝ^{B×T×F}
                x = x.reshape(-1, self.num_forecast_steps, self.out_features)
            else:
                x = self.residual_norm(x + projected_residual)
            
        # Apply output activation
        # Non-linear transformation ψ
        if self.output_activation:
            x = self.output_activation(x)
            
        if return_intermediates:
            return x, intermediates
        return x
        
        
class QuantileOutputNetwork(DynamicOutputNetwork):
    """ Output network for quantile regression """
    
    # Defines input dimension, quantity of quantiles and forecast steps
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
