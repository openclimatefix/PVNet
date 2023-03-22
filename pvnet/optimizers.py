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