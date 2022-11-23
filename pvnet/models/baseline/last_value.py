import logging

from nowcasting_dataloader.batch import BatchML

from pvnet.models.base_model import BaseModel

from ocf_datapipes.utils.consts import BatchKey

logging.basicConfig()
_LOG = logging.getLogger("pvnet")
_LOG.setLevel(logging.DEBUG)


class Model(BaseModel):
    name = "last_value"

    def __init__(
        self,
        forecast_minutes: int = 12,
        history_minutes: int = 6,
        output_variable="pv_yield",
    ):
        """
        Simple baseline model that takes the last pv yield value and copies it forward
        """

        self.forecast_minutes = forecast_minutes
        self.history_minutes = history_minutes
        self.output_variable = output_variable

        super().__init__()

    def forward(self, x: dict):

        # Shape: batch_size, seq_length, n_sites
        if self.output_variable == "gsp_yield":
            gsp_yield = x[BatchKey.gsp]
        else:
            gsp_yield = x[BatchKey.pv]

        # take the last value non forecaster value and the first in the pv yeild
        # (this is the pv site we are preditcting for)
        y_hat = gsp_yield[:, -self.forecast_len - 1, 0]

        # expand the last valid forward n predict steps
        out = y_hat.unsqueeze(1).repeat(1, self.forecast_len)
        # shape: batch_size, forecast_len

        return out
