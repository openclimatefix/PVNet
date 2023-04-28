import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch import _VF
import warnings

    
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
    """
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
    """Residual block of 'full pre-activation' from figure 4(e) of [1]. This was the best performing 
    residual block tested in [1].
    
    Sources:
        [1] https://arxiv.org/pdf/1603.05027.pdf
        
    Args:
        in_features: Number of input features.
        n_layers: Number of layers in residual pathway.
    """
    def __init__(
        self,
        in_features: int,
        n_layers: int = 2,
        dropout_frac: float = 0.,
    ):
        
        super().__init__()

        layers = []
        for i in range(n_layers):
            layers += [
                nn.BatchNorm1d(in_features),
                nn.Dropout(p=dropout_frac),
                nn.LeakyReLU(),
                nn.Linear(
                    in_features=in_features, 
                    out_features=in_features,
                ),
                
            ]

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)+x
    
    
class ImageEmbedding(nn.Module):

    def __init__(
        self, 
        num_embeddings, 
        image_size_pixels,
        sequence_length,
        padding_idx=None, 
        max_norm=None, 
        norm_type=2.0, 
        scale_grad_by_freq=False, 
        sparse=False, 
        _weight=None, 
        _freeze=False, 
        device=None, 
        dtype=None
    ):
        super().__init__()
        self.image_size_pixels = image_size_pixels
        self.sequence_length = sequence_length
        self._embed = nn.Embedding(
            num_embeddings=num_embeddings, 
            embedding_dim=image_size_pixels*image_size_pixels,
            padding_idx=padding_idx,
            max_norm=max_norm, 
            norm_type=norm_type, 
            scale_grad_by_freq=scale_grad_by_freq, 
            sparse=sparse, 
            _weight=_weight, 
            _freeze=_freeze, 
            device=device, 
            dtype=dtype,
        )
        
    def forward(self, x, id):
        emb = self._embed(id)
        emb = emb.reshape((-1, 1, 1, self.image_size_pixels, self.image_size_pixels))
        emb = emb.repeat(1, 1, self.sequence_length, 1, 1)
        x = torch.cat((x, emb), dim=1)
        return x
        

class CompleteDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def forward(self, x):
        if np.random.uniform()<self.p:
            return torch.zeros_like(x)
        else:
            return x


        
class CompleteDropoutNd(nn.Module):
    
    __constants__ = ['p', 'inplace', 'n_dim']
    p: float
    inplace: bool
    n_dim: int
    
    def __init__(self, n_dim, p=0.5, inplace=False):
        
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace
        self.n_dim = n_dim
    
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Randomly zero out all channels (a channel is a 3D feature map,
        e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
        batched input is a 3D tensor :math:`\text{input}[i, j]`) of the input tensor).
        Each channel will be zeroed out independently on every forward call with
        probability :attr:`p` using samples from a Bernoulli distribution.

        See :class:`~torch.nn.Dropout3d` for details.

        Args:
            p: probability of a channel to be zeroed. Default: 0.5
            training: apply dropout if is ``True``. Default: ``True``
            inplace: If set to ``True``, will do this operation in-place. Default: ``False``
        """
        
        p = self.p
        inp_dim = input.dim()

        if inp_dim not in (self.n_dim+1, self.n_dim+2):
            warn_msg = (f"dropoutNd: Received a {inp_dim}-D input to dropout3d, which is deprecated "
                        "and will result in an error in a future release. To retain the behavior "
                        "and silence this warning, please use dropout instead. Note that dropout3d "
                        "exists to provide channel-wise dropout on inputs with 3 spatial dimensions, "
                        "a channel dimension, and an optional batch dimension (i.e. 4D or 5D inputs).")
            warnings.warn(warn_msg)

        is_batched = inp_dim == self.n_dim+2
        if not is_batched:
            input = input.unsqueeze_(0) if self.inplace else input.unsqueeze(0)

        input = input.unsqueeze_(1) if self.inplace else input.unsqueeze(1)

        result = (
            _VF.feature_dropout_(input, p,  self.training) if self.inplace 
            else _VF.feature_dropout(input, p, self.training)
        )
        
        result = result.squeeze_(1) if self.inplace else result.squeeze(1)

        if not is_batched:
            result = result.squeeze_(0) if self.inplace else result.squeeze(0)
            
        return result

