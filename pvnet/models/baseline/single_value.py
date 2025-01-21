"""Average value model"""
import torch
from torch import nn

import pvnet
from pvnet.models.base_model import BaseModel
from pvnet.optimizers import AbstractOptimizer


class Model(BaseModel):
    """Simple baseline model that predicts always the same value."""

    name = "single_value"

    def __init__(
        self,
        forecast_minutes: int = 120,
        history_minutes: int = 60,
        optimizer: AbstractOptimizer = pvnet.optimizers.Adam(),
    ):
        """Simple baseline model that predicts always the same value.

        Args:
            history_minutes (int): Length of the GSP history period in minutes
            forecast_minutes (int): Length of the GSP forecast period in minutes
            optimizer (AbstractOptimizer): Optimizer
        """
        super().__init__(history_minutes, forecast_minutes, optimizer)
        self._value = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.save_hyperparameters()

    def forward(self, x: dict):
        """Run model forward on dict batch of data"""
        # Returns a single value at all steps
        y_hat = torch.zeros_like(x["gsp"][:, : self.forecast_len]) + self._value
        return y_hat
