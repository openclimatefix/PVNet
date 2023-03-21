from ocf_datapipes.utils.consts import BatchKey
from pvnet.models.base_model import BaseModel


class Model(BaseModel):
    name = "last_value"

    def __init__(
        self,
        forecast_minutes: int = 12,
        history_minutes: int = 6,
    ):
        """
        Simple baseline model that takes the last gsp yield value and copies it forward
        """
        self.forecast_minutes = forecast_minutes
        self.history_minutes = history_minutes
        super().__init__()

    def forward(self, x: dict):

        # Shape: batch_size, seq_length, n_sites
        gsp_yield = x[BatchKey.gsp]

        # take the last value non forecaster value and the first in the pv yeild
        # (this is the pv site we are preditcting for)
        y_hat = gsp_yield[:, -self.forecast_len - 1, 0]

        # expand the last valid forward n predict steps
        out = y_hat.unsqueeze(1).repeat(1, self.forecast_len)
        # shape: batch_size, forecast_len

        return out
