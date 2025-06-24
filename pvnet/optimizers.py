"""Optimizer factory-function classes.
"""

from abc import ABC, abstractmethod

import math
import torch
import torch.nn as nn


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


def get_layerwise_parameters(model, lr_scaling_factor=0.1):
    param_groups = []
    assigned_params = set()

    embedding_params = []
    for name, param in model.named_parameters():
        is_embedding = False
        parts = name.split('.')
        current_module = model
        for part in parts[:-1]:
            if hasattr(current_module, part):
                current_module = getattr(current_module, part)
            else:
                current_module = None
                break
        if current_module and isinstance(current_module, nn.Embedding):
            embedding_params.append(param)
            assigned_params.add(param)
    
    if embedding_params:
        param_groups.append({
            "params": embedding_params,
            "weight_decay": 0.0,
            "lr_scale": lr_scaling_factor ** 0
        })

    depth_groups = {}
    for name, param in model.named_parameters():
        if param in assigned_params:
            continue

        depth = name.count('.')
        
        if depth not in depth_groups:
            depth_groups[depth] = []
        depth_groups[depth].append(param)
        assigned_params.add(param)

    sorted_depths = sorted(depth_groups.keys())
    
    for depth in sorted_depths:
        lr_scale = lr_scaling_factor ** depth 
        
        param_groups.append({
            "params": depth_groups[depth],
            "lr_scale": lr_scale,
        })

    return param_groups


def compute_adaptive_weight_decay(
    initial_wd: float,
    final_wd: float,
    schedule_type: str,
    current_epoch: int,
    total_epochs: int,
) -> float:
    if total_epochs == 0:
        return final_wd
    
    progress = current_epoch / total_epochs 

    if schedule_type == "cosine":
        return final_wd + 0.5 * (initial_wd - final_wd) * (1 + math.cos(math.pi * progress))
    elif schedule_type == "linear":
        return initial_wd * (1 - progress) + final_wd * progress
    elif schedule_type == "exponential":
        if initial_wd == 0: return final_wd
        return initial_wd * ((final_wd / initial_wd) ** progress)
    else:
        raise ValueError(f"Unknown weight decay schedule type: {schedule_type}")


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


class EmbAdamWLayerwiseAdaptive(AbstractOptimizer):
    def __init__(
        self,
        lr: float = 0.0001,
        lr_scaling_factor: float = 0.1,
        wd_initial: float = 0.25,
        wd_final: float = 0.1,
        wd_schedule: str = "cosine",
        patience: int = 3,
        factor: float = 0.5,
        threshold: float = 2e-4,
        **opt_kwargs,
    ):
        super().__init__()
        self._lr = lr
        self.lr_scaling_factor = lr_scaling_factor
        self.wd_initial = wd_initial
        self.wd_final = wd_final
        self.wd_schedule = wd_schedule
        self.patience = patience
        self.factor = factor
        self.threshold = threshold
        self.opt_kwargs = opt_kwargs
        self.current_wd = wd_initial
        self.num_epochs = None
        self.optimizer = None

    def __call__(self, model: torch.nn.Module):
        param_groups_info = get_layerwise_parameters(model, self.lr_scaling_factor)

        optim_groups = []
        for group_info in param_groups_info:
            group = {
                "params": group_info["params"],
                "lr": self._lr * group_info["lr_scale"],
                "weight_decay": group_info.get("weight_decay", self.wd_initial)
            }
            optim_groups.append(group)
            
        opt = torch.optim.AdamW(optim_groups, **self.opt_kwargs)
        self.optimizer = opt
        
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

    def update_weight_decay(self, epoch: int, total_epochs: int):
        self.num_epochs = total_epochs
        new_wd = compute_adaptive_weight_decay(
            self.wd_initial, self.wd_final, self.wd_schedule, epoch, total_epochs
        )
        self.current_wd = new_wd
        
        if self.optimizer is None:
            return

        for param_group in self.optimizer.param_groups:
            if param_group.get("weight_decay") != 0.0:
                param_group["weight_decay"] = new_wd
