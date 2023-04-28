import torch
from torch import nn
from abc import ABCMeta, abstractmethod

from pvnet.models.base_model import BaseModel
from torchvision.transforms import CenterCrop
from collections import OrderedDict

from pvnet.models.conv3d.basic_blocks import ResidualLinearBlock, ResidualLinearBlock2


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
        
    def cat_modes(self, x):
        if isinstance(x, OrderedDict):
            return torch.cat([value for key, value in x.items()], dim=1)
        elif isinstance(x, torch.Tensor):
            return x
        else:
            raise ValueError(f"Input of unexpected type {type(x)}")
        
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
            nn.ELU(),
            nn.Linear(in_features=fc_hidden_features, out_features=out_features),
            nn.ReLU(),
        )


    def forward(self, x):
        x = self.cat_modes(x)
        return self.model(x)
    
    
class TabNet(AbstractTabularNetwork):
    """
    An implmentation of TabNet which also uses the default FC network in parallel. The FC network 
    was included to add skip connections and stabalize the training.

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

        from pytorch_tabnet.tab_network import TabNet as _TabNetModel

        super().__init__(in_features, out_features)
        
        self._tabnet = _TabNetModel(
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
        
        self._simple_model = DefaultFCNet(
            in_features=in_features,
            out_features=out_features,
            fc_hidden_features=32,
        )
        
        self.activation = nn.LeakyReLU(negative_slope=0.01)
    
    def forward(self, x):
        # TODO: USE THIS LOSS COMPONENT
        #loss = self.compute_loss(output, y)
        # Add the overall sparsity loss
        #loss = loss - self.lambda_sparse * M_loss
        x = self.cat_modes(x)
        out1, M_loss = self._tabnet(x)
        out2 = self._simple_model(x)
        return self.activation(out1+out2)


    
class ResFCNet(AbstractTabularNetwork):
    """
    Fully connected deep network based on ResNet architecture. With n_res_blocks=0 this becomes
    equivalent to `DefaultFCNet`.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        fc_hidden_features: Number of features in middle hidden layers.
        n_res_blocks: Number of residual blocks to use.
        res_block_layers: Number of fully-connected layers used in each residual block.
        dropout_frac: Probability of an element to be zeroed in the residual pathways.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        fc_hidden_features: int = 128,
        n_res_blocks: int = 4,
        res_block_layers: int = 2,
        dropout_frac: float = 0.2,
    ):
        

        super().__init__(in_features, out_features)
                
        model = [
            nn.Linear(in_features=in_features, out_features=fc_hidden_features),
        ]
        
        for i in range(n_res_blocks):
            model += [
                ResidualLinearBlock(
                    in_features=fc_hidden_features,
                    n_layers=res_block_layers,
                    dropout_frac=dropout_frac,
                )
            ]
                
        model += [
            nn.ELU(),
            nn.Linear(in_features=fc_hidden_features, out_features=out_features),
            nn.LeakyReLU(negative_slope=0.01),
        ]
        self.model = nn.Sequential(*model)


    def forward(self, x):
        x = self.cat_modes(x)
        return self.model(x)
    
class ResFCNet2(AbstractTabularNetwork):
    """
    Fully connected deep network based on ResNet architecture. With n_res_blocks=0 this becomes
    equivalent to `DefaultFCNet`.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        fc_hidden_features: Number of features in middle hidden layers.
        n_res_blocks: Number of residual blocks to use.
        res_block_layers: Number of fully-connected layers used in each residual block.
        dropout_frac: Probability of an element to be zeroed in the residual pathways.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        fc_hidden_features: int = 128,
        n_res_blocks: int = 4,
        res_block_layers: int = 2,
        dropout_frac=0.,
        **kwargs,
    ):
        

        super().__init__(in_features, out_features)
                
        model = [
            nn.Linear(in_features=in_features, out_features=fc_hidden_features),
        ]
        
        for i in range(n_res_blocks):
            model += [
                ResidualLinearBlock2(
                    in_features=fc_hidden_features,
                    n_layers=res_block_layers,
                    dropout_frac=dropout_frac,
                )
            ]
                
        model += [
            nn.LeakyReLU(),
            nn.Linear(in_features=fc_hidden_features, out_features=out_features),
            nn.LeakyReLU(negative_slope=0.01),
        ]
        
        self.model = nn.Sequential(*model)


    def forward(self, x):
        x = self.cat_modes(x)
        return self.model(x)
    
    
class SNN(nn.Module):
    """Self normalising neural network implementation borrowed from [1] and proposed in [2].
    
    Sources
    -------
        [1] https://github.com/tonyduan/snn/blob/master/snn/models.py
        [2] https://arxiv.org/pdf/1706.02515v5.pdf
    """

    def __init__(self, in_dim, out_dim, hidden_dim, n_layers, dropout_prob=0.0):
        super().__init__()
        layers = OrderedDict()
        for i in range(n_layers - 1):
            if i == 0:
                layers[f"fc{i}"] = nn.Linear(in_dim, hidden_dim, bias=False)
            else:
                layers[f"fc{i}"] = nn.Linear(hidden_dim, hidden_dim, bias=False)
            layers[f"selu_{i}"] = nn.SELU()
            layers[f"dropout_{i}"] = nn.AlphaDropout(p=dropout_prob)
        layers[f"fc_{i+1}"] = nn.Linear(hidden_dim, out_dim, bias=True)
        self.network = nn.Sequential(layers)
        self.reset_parameters()

    def forward(self, x):
        x = self.cat_modes(x)
        return self.network(x)

    def reset_parameters(self):
        for layer in self.network:
            if not isinstance(layer, nn.Linear):
                continue
            nn.init.normal_(layer.weight, std=1 / math.sqrt(layer.out_features))
            if layer.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(layer.bias, -bound, bound)

    

  