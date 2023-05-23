"""Linear networks used for the fusion model"""
from torch import nn

from pvnet.models.multimodal.linear_networks.basic_blocks import (
    AbstractLinearNetwork,
    ResidualLinearBlock,
    ResidualLinearBlock2,
)


class DefaultFCNet(AbstractLinearNetwork):
    """Similar to the original FCNet module used in PVNet, with a few minor tweaks.

    This is a 2-layer fully connected block, with internal ELU activations and output ReLU.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        fc_hidden_features: int = 128,
    ):
        """Similar to the original FCNet module used in PVNet, with a few minor tweaks.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            fc_hidden_features: Number of features in middle hidden layer.
        """
        super().__init__(in_features, out_features)

        self.model = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=fc_hidden_features),
            nn.ELU(),
            nn.Linear(in_features=fc_hidden_features, out_features=out_features),
            nn.ReLU(),
        )

    def forward(self, x):
        """Run model forward"""
        x = self.cat_modes(x)
        return self.model(x)


class ResFCNet(AbstractLinearNetwork):
    """Fully-connected deep network based on ResNet architecture.

    Internally, this network uses ELU activations throughout the residual blocks.
    With n_res_blocks=0 this becomes equivalent to `DefaultFCNet`.
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
        """Fully-connected deep network based on ResNet architecture.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            fc_hidden_features: Number of features in middle hidden layers.
            n_res_blocks: Number of residual blocks to use.
            res_block_layers: Number of fully-connected layers used in each residual block.
            dropout_frac: Probability of an element to be zeroed in the residual pathways.
        """
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
        """Run model forward"""
        x = self.cat_modes(x)
        return self.model(x)


class ResFCNet2(AbstractLinearNetwork):
    """Fully connected deep network based on ResNet architecture.

    This architecture is similar to
    `ResFCNet`, except that it uses LeakyReLU activations internally, and batchnorm in the residual
    branches. The residual blocks are implemented based on the best performing block in [1].

    Sources:
        [1] https://arxiv.org/pdf/1603.05027.pdf
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        fc_hidden_features: int = 128,
        n_res_blocks: int = 4,
        res_block_layers: int = 2,
        dropout_frac=0.0,
    ):
        """Fully connected deep network based on ResNet architecture.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            fc_hidden_features: Number of features in middle hidden layers.
            n_res_blocks: Number of residual blocks to use.
            res_block_layers: Number of fully-connected layers used in each residual block.
            dropout_frac: Probability of an element to be zeroed in the residual pathways.
        """
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
        """Run model forward"""
        x = self.cat_modes(x)
        return self.model(x)


class SNN(AbstractLinearNetwork):
    """Self normalising neural network implementation borrowed from [1] and proposed in [2].

    Sources:
        [1] https://github.com/tonyduan/snn/blob/master/snn/models.py
        [2] https://arxiv.org/pdf/1706.02515v5.pdf

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        fc_hidden_features: Number of features in middle hidden layers.
        n_layers: Number of fully-connected layers used in the network.
        dropout_frac: Probability of an element to be zeroed.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        fc_hidden_features: int = 128,
        n_layers: int = 10,
        dropout_frac: float = 0.0,
    ):
        """Self normalising neural network implementation borrowed from [1] and proposed in [2].

        Sources:
            [1] https://github.com/tonyduan/snn/blob/master/snn/models.py
            [2] https://arxiv.org/pdf/1706.02515v5.pdf

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            fc_hidden_features: Number of features in middle hidden layers.
            n_layers: Number of fully-connected layers used in the network.
            dropout_frac: Probability of an element to be zeroed.

        """
        super().__init__(in_features, out_features)

        layers = [
            nn.Linear(in_features, fc_hidden_features, bias=False),
            nn.SELU(),
            nn.AlphaDropout(p=dropout_frac),
        ]
        for i in range(1, n_layers - 1):
            layers += [
                nn.Linear(fc_hidden_features, fc_hidden_features, bias=False),
                nn.SELU(),
                nn.AlphaDropout(p=dropout_frac),
            ]
        layers += [
            nn.Linear(fc_hidden_features, out_features, bias=True),
            nn.LeakyReLU(negative_slope=0.01),
        ]

        self.network = nn.Sequential(*layers)
        self._reset_parameters()

    def forward(self, x):
        """Run model forward"""
        x = self.cat_modes(x)
        return self.network(x)

    def _reset_parameters(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=layer.out_features**-0.5)
                if layer.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                    bound = fan_in**-0.5
                    nn.init.uniform_(layer.bias, -bound, bound)


class TabNet(AbstractLinearNetwork):
    """An implmentation of TabNet [1].

    The implementation comes rom `pytorch_tabnet` and this must be installed for use.


    Sources:
        [1] https://arxiv.org/abs/1908.07442
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
        """An implmentation of TabNet [1].

        Sources:
            [1] https://arxiv.org/abs/1908.07442

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

        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        """Run model forward"""
        # TODO: USE THIS LOSS COMPONENT
        # loss = self.compute_loss(output, y)
        # Add the overall sparsity loss
        # loss = loss - self.lambda_sparse * M_loss
        x = self.cat_modes(x)
        out1, M_loss = self._tabnet(x)
        return self.activation(out1)
