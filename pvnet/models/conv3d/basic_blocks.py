import torch
from torch import nn
import torch.nn.functional as F

    
class ResidualConv3dBlock(nn.Module):
    """Residual block of 'full pre-activation' from figure 4(e) of [1]. This was the best performing 
    residual block tested in [1].
    
    Sources:
        [1] https://arxiv.org/pdf/1603.05027.pdf
        
    Args:
        in_features: Number of input features.
        n_layers: Number of layers in residual pathway.
        dropout_frac: Probability of an element to be zeroed.
    """
    def __init__(
        self,
        in_channels,
        n_layers: int = 2,
        dropout_frac: float = 0.0,
    ):
        
        super().__init__()

        layers = []
        for i in range(n_layers):
            layers += [
                #nn.BatchNorm3d(in_channels),
                nn.ELU(),
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=(3, 3, 3),
                    padding=(1, 1, 1),
                )
            ]
            if dropout_frac>0:
                layers+=[nn.Dropout3d(p=dropout_frac)]    

        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)+x

    
    
class ResidualLinearBlock(nn.Module):
    """Residual block of 'full pre-activation' from figure 4(e) of [1]. This was the best performing 
    residual block tested in [1].
    
    Sources:
        [1] https://arxiv.org/pdf/1603.05027.pdf
        
    Args:
        in_features: Number of input features.
        n_layers: Number of layers in residual pathway.
        dropout_frac: Probability of an element to be zeroed.
    """
    def __init__(
        self,
        in_features: int,
        n_layers: int = 2,
        dropout_frac: float = 0.0,
    ):
        
        super().__init__()

        layers = []
        for i in range(n_layers):
            layers += [
                #nn.BatchNorm1d(in_features),
                nn.ELU(),
                nn.Linear(
                    in_features=in_features, 
                    out_features=in_features,
                ),
            ]
            if dropout_frac>0:
                layers+=[nn.Dropout(p=dropout_frac)]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)+x
    
class ResidualLinearBlock2(nn.Module):
    """Residual block
        
    Args:
        in_features: Number of input features.
        n_layers: Number of layers in residual pathway.
    """
    def __init__(
        self,
        in_features: int,
        n_layers: int = 2,
    ):
        
        super().__init__()

        layers = []
        for i in range(n_layers):
            layers += [
                nn.Linear(
                    in_features=in_features, 
                    out_features=in_features,
                ),
                nn.BatchNorm1d(in_features)
            ]
            if i!=(n_layers-1):
                layers += [nn.ReLU()]

        self.model = nn.Sequential(*layers)
        self.final_act = nn.ReLU()
    
    def forward(self, x):
        return self.final_act(self.model(x)+x)
