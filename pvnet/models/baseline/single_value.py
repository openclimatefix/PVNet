from torch import nn
import torch

from pvnet.models.base_model import BaseModel
from pvnet.optimizers import AbstractOptimizer
import pvnet

from ocf_datapipes.utils.consts import BatchKey


class Model(BaseModel):
    """Simple baseline model that predicts always the same value. Mainly used for testing.
    """
        
    name = "single_value"

    def __init__(
        self,
        forecast_minutes: int = 120,
        history_minutes: int = 60,
        optimizer: AbstractOptimizer = pvnet.optimizers.Adam(),
    ):
        super().__init__(history_minutes, forecast_minutes, optimizer)
        self._value = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.save_hyperparameters()

    def forward(self, x: dict):
        # Returns a single value at all steps
        y_hat = torch.zeros_like(x[BatchKey.gsp][:, :self.forecast_len, 0]) + self._value
        return y_hat