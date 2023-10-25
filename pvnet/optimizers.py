"""Optimizer factory-function classes.
"""

from abc import ABC, abstractmethod

import torch


class AbstractOptimizer(ABC):
    """Abstract class for optimizer

    Optimizer classes will be used by model like:
    > OptimizerGenerator = AbstractOptimizer()
    > optimizer = OptimizerGenerator(model)
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

    def __call__(self, model):
        """Return optimizer"""
        return torch.optim.Adam(model.parameters(), lr=self.lr, **self.kwargs)


class AdamW(AbstractOptimizer):
    """AdamW optimizer"""

    def __init__(self, lr=0.0005, **kwargs):
        """AdamW optimizer"""
        self.lr = lr
        self.kwargs = kwargs

    def __call__(self, model):
        """Return optimizer"""
        return torch.optim.AdamW(model.parameters(), lr=self.lr, **self.kwargs)


def find_submodule_parameters(model, search_modules):
    """Finds all parameters within given submodule types

    Args:
        model: torch Module to search through
        search_modules: List of submodule types to search for
    """
    if isinstance(model, search_modules):
        return model.parameters()

    children = list(model.children())
    if len(children) == 0:
        return []
    else:
        params = []
        for c in children:
            params += find_submodule_parameters(c, search_modules)
        return params


def find_other_than_submodule_parameters(model, ignore_modules):
    """Finds all parameters not with given submodule types

    Args:
        model: torch Module to search through
        ignore_modules: List of submodule types to ignore
    """
    if isinstance(model, ignore_modules):
        return []

    children = list(model.children())
    if len(children) == 0:
        return model.parameters()
    else:
        params = []
        for c in children:
            params += find_other_than_submodule_parameters(c, ignore_modules)
        return params


class EmbAdamWReduceLROnPlateau(AbstractOptimizer):
    """AdamW optimizer and reduce on plateau scheduler"""

    def __init__(
        self, lr=0.0005, weight_decay=0.01, patience=3, factor=0.5, threshold=2e-4, **opt_kwargs
    ):
        """AdamW optimizer and reduce on plateau scheduler"""
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.factor = factor
        self.threshold = threshold
        self.opt_kwargs = opt_kwargs

    def __call__(self, model):
        """Return optimizer"""

        search_modules = (torch.nn.Embedding,)

        no_decay = find_submodule_parameters(model, search_modules)
        decay = find_other_than_submodule_parameters(model, search_modules)

        optim_groups = [
            {"params": decay, "weight_decay": self.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        opt = torch.optim.AdamW(optim_groups, lr=self.lr, **self.opt_kwargs)

        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=self.factor,
            patience=self.patience,
            threshold=self.threshold,
        )
        sch = {
            "scheduler": sch,
            "monitor": "quantile_loss/val" if model.use_quantile_regression else "MAE/val",
        }
        return [opt], [sch]


class AdamWReduceLROnPlateau(AbstractOptimizer):
    """AdamW optimizer and reduce on plateau scheduler"""

    def __init__(
        self, lr=0.0005, patience=3, factor=0.5, threshold=2e-4, step_freq=None, **opt_kwargs
    ):
        """AdamW optimizer and reduce on plateau scheduler"""
        self._lr = lr
        self.patience = patience
        self.factor = factor
        self.threshold = threshold
        self.step_freq = step_freq
        self.opt_kwargs = opt_kwargs

    def _call_multi(self, model):
        remaining_params = {k: p for k, p in model.named_parameters()}

        group_args = []

        for key in self._lr.keys():
            if key == "default":
                continue

            submodule_params = []
            for param_name in list(remaining_params.keys()):
                if param_name.startswith(key):
                    submodule_params += [remaining_params.pop(param_name)]

            group_args += [{"params": submodule_params, "lr": self._lr[key]}]

        remaining_params = [p for k, p in remaining_params.items()]
        group_args += [{"params": remaining_params}]

        opt = torch.optim.AdamW(
            group_args, lr=self._lr["default"] if model.lr is None else model.lr, **self.opt_kwargs
        )
        sch = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                factor=self.factor,
                patience=self.patience,
                threshold=self.threshold,
            ),
            "monitor": "quantile_loss/val" if model.use_quantile_regression else "MAE/val",
        }

        return [opt], [sch]

    def __call__(self, model):
        """Return optimizer"""
        if not isinstance(self._lr, float):
            return self._call_multi(model)
        else:
            assert False
            default_lr = self._lr if model.lr is None else model.lr
            opt = torch.optim.AdamW(model.parameters(), lr=default_lr, **self.opt_kwargs)
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                factor=self.factor,
                patience=self.patience,
                threshold=self.threshold,
            )
            sch = {
                "scheduler": sch,
                "monitor": "quantile_loss/val" if model.use_quantile_regression else "MAE/val",
            }
            return [opt], [sch]
