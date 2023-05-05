"""Optimizer factory-function classes.
"""

from abc import ABC, abstractmethod

import torch


class AbstractOptimizer(ABC):
    """Optimizer classes will be used by model like:
    > OptimizerGenerator = AbstractOptimizer()
    > optimizer = OptimizerGenerator(model.parameters())
    The returned object `optimizer` must be something that may be returned by `pytorch_lightning`'s
    `configure_optimizers()` method.
    See :
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers

    """

    @abstractmethod
    def __call__(self):
        pass


class Adam(AbstractOptimizer):
    def __init__(self, lr=0.0005, **kwargs):
        self.lr = lr
        self.kwargs = kwargs

    def __call__(self, model_parameters):
        return torch.optim.Adam(model_parameters, lr=self.lr, **self.kwargs)


class AdamW(AbstractOptimizer):
    def __init__(self, lr=0.0005, **kwargs):
        self.lr = lr
        self.kwargs = kwargs

    def __call__(self, model_parameters):
        return torch.optim.AdamW(model_parameters, lr=self.lr, **self.kwargs)


class AdamWReduceLROnPlateau(AbstractOptimizer):
    def __init__(self, lr=0.0005, patience=3, factor=0.5, threshold=2e-4, **opt_kwargs):
        self.lr = lr
        self.patience = patience
        self.factor = factor
        self.threshold = threshold
        self.opt_kwargs = opt_kwargs

    def __call__(self, model_parameters):
        opt = torch.optim.AdamW(model_parameters, lr=self.lr, **self.opt_kwargs)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=self.factor,
            patience=self.patience,
            threshold=self.threshold,
        )
        sch = {"scheduler": sch, "monitor": "MAE/train"}
        return [opt], [sch]
