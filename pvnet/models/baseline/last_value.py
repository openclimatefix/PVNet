"""Persistence model"""

from ocf_datapipes.batch import BatchKey

import pvnet
from pvnet.models.base_model import BaseModel
from pvnet.optimizers import AbstractOptimizer


class Model(BaseModel):
    """Simple baseline model that takes the last gsp yield value and copies it forward."""

    name = "last_value"

    def __init__(
        self,
        forecast_minutes: int = 12,
        history_minutes: int = 6,
        optimizer: AbstractOptimizer = pvnet.optimizers.Adam(),
    ):
        """Simple baseline model that takes the last gsp yield value and copies it forward.

        Args:
            history_minutes (int): Length of the GSP history period in minutes
            forecast_minutes (int): Length of the GSP forecast period in minutes
            optimizer (AbstractOptimizer): Optimizer
        """

        super().__init__(history_minutes, forecast_minutes, optimizer)
        self.save_hyperparameters()

    def forward(self, x: dict):
        """Run model forward on dict batch of data"""
        # Shape: batch_size, seq_length, n_sites
        gsp_yield = x[BatchKey.gsp]

        # take the last value non forecaster value and the first in the pv yeild
        # (this is the pv site we are preditcting for)
        y_hat = gsp_yield[:, -self.forecast_len - 1, 0]

        # expand the last valid forward n predict steps
        out = y_hat.unsqueeze(1).repeat(1, self.forecast_len)
        return out
