import torch
from torch import nn
from abc import ABCMeta, abstractmethod

from pvnet.models.base_model import BaseModel
from torchvision.transforms import CenterCrop
from pytorch_tabnet.tab_network import TabNet as _TabNetModel


class AbstractTabularNetwork(nn.Module, metaclass=ABCMeta):
    """Abstract class for a network to combine the features from all the inputs.
    
    Args:
        in_features: Number of input features.
        out_features: Number of output features.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        super().__init__()
        
    @abstractmethod
    def forward(self):
        pass
        
    
class DefaultFCNet(AbstractTabularNetwork):
    """
    Similar to the original FCNet module used in PVNet, with a few minor tweaks.
    A 2-layer 

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        fc_hidden_features: Number of features in middle hidden layer
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        fc_hidden_features: int=128,
    ):


        super().__init__(in_features, out_features)
    
        self.model = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=fc_hidden_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=fc_hidden_features, out_features=out_features),
            nn.ReLU(),
        )


    def forward(self, x):
        return self.model(x)
    
    
class TabNet(AbstractTabularNetwork):
    """
    Defines TabNet network

    Args:
        in_features: int
            Number of input features.
        out_features: int
            Number of output features.        
        n_d : int
            Dimension of the prediction  layer (usually between 4 and 64)
        n_a : int
            Dimension of the attention  layer (usually between 4 and 64)
        n_steps : int
            Number of successive steps in the network (usually between 3 and 10)
        gamma : float
            Float above 1, scaling factor for attention updates (usually between 1.0 to 2.0)
        cat_idxs : list of int
            Index of each categorical column in the dataset
        cat_dims : list of int
            Number of categories in each categorical column
        cat_emb_dim : int or list of int
            Size of the embedding of categorical features
            if int, all categorical features will have same embedding size
            if list of int, every corresponding feature will have specific size
        n_independent : int
            Number of independent GLU layer in each GLU block (default 2)
        n_shared : int
            Number of independent GLU layer in each GLU block (default 2)
        epsilon : float
            Avoid log(0), this should be kept very low
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        cat_idxs=[],
        cat_dims=[],
        cat_emb_dim=1,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
    ):


        super().__init__(in_features, out_features)
        
        self.model = _TabNetModel(
            input_dim=in_features,
            output_dim=out_features,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=cat_emb_dim,
            n_independent=n_independent,
            n_shared=n_shared,
            epsilon=epsilon,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
            mask_type=mask_type,
        )
        self.activation = nn.PRELU()
    
    def forward(self, x):
        return self.activation(self.model(x))
  