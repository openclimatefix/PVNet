"""Optimizer factory-function classes.
"""

from abc import ABC, abstractmethod

import torch


class AbstractOptimizer(ABC):
    """Abstract class for optimizer

    Optimizer classes will be used by model like:
    > OptimizerGenerator = AbstractOptimizer()
    > optimizer = OptimizerGenerator(model.parameters())
    The returned object `optimizer` must be something that may be returned by `pytorch_lightning`'s
    `configure_optimizers()` method.
    See :
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers

    """

    @abstractmethod
    def __call__(self):
        """Abstract call"""
        pass


class Adam(AbstractOptimizer):
    """Adam optimizer"""

    def __init__(self, lr=0.0005, **kwargs):
        """Adam optimizer"""
        self.lr = lr
        self.kwargs = kwargs

    def __call__(self, model_parameters):
        """Return optimizer"""
        return torch.optim.Adam(model_parameters, lr=self.lr, **self.kwargs)


class AdamW(AbstractOptimizer):
    """AdamW optimizer"""

    def __init__(self, lr=0.0005, **kwargs):
        """AdamW optimizer"""
        self.lr = lr
        self.kwargs = kwargs

    def __call__(self, model_parameters):
        """Return optimizer"""
        return torch.optim.AdamW(model_parameters, lr=self.lr, **self.kwargs)


class AdamWReduceLROnPlateau(AbstractOptimizer):
    """AdamW optimizer and reduce on plateau scheduler"""

    def __init__(self, lr=0.0005, patience=3, factor=0.5, threshold=2e-4, **opt_kwargs):
        """AdamW optimizer and reduce on plateau scheduler"""
        self.lr = lr
        self.patience = patience
        self.factor = factor
        self.threshold = threshold
        self.opt_kwargs = opt_kwargs

    def __call__(self, model_parameters):
        """Return optimizer"""
        opt = torch.optim.AdamW(model_parameters, lr=self.lr, **self.opt_kwargs)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=self.factor,
            patience=self.patience,
            threshold=self.threshold,
        )
        sch = {"scheduler": sch, "monitor": "MAE/train"}
        return [opt], [sch]
