"""Architecture for simple learned weighted average of the downwards short wave radiation flux"""
from typing import Optional

import torch
from ocf_datapipes.batch import BatchKey
from torch import nn

import pvnet
from pvnet.models.base_model import BaseModel
from pvnet.optimizers import AbstractOptimizer


class Model(BaseModel):
    """Model that learns an linear interpolation of NWP dwsrf to predict output.

    This model learns to compute a weighted average of the downward short-wave radiation flux for
    each GSP. The same averaging is used for each step in the NWP input sequence. It also learns
    a linear time-interpolation scheme to map between the NWP-step weighted average and the
    predicted GSP output.

    """

    name = "nwp_weighting"

    def __init__(
        self,
        forecast_minutes: int = 30,
        history_minutes: int = 60,
        nwp_image_size_pixels: int = 64,
        nwp_forecast_minutes: Optional[int] = None,
        nwp_history_minutes: Optional[int] = None,
        dwsrf_channel: int = 0,
        optimizer: AbstractOptimizer = pvnet.optimizers.Adam(),
    ):
        """Model that learns an linear interpolation of NWP dwsrf to predict output.

        Args:
            forecast_minutes: The amount of minutes that should be forecasted.
            history_minutes: The default amount of historical minutes that are used.
            nwp_forecast_minutes: Period of future NWP forecast data to use. Defaults to
                `forecast_minutes` if not provided.
            nwp_history_minutes: Period of historical data to use for NWP data. Defaults to
                `history_minutes` if not provided.
            nwp_image_size_pixels: Image size (assumed square) of the NWP data.
            dwsrf_channel: Which index of the NWP input is the dwsrf channel.
            optimizer: Optimizer factory function used for network.
        """
        super().__init__(history_minutes, forecast_minutes, optimizer)

        self.dwsrf_channel = dwsrf_channel

        if nwp_history_minutes is None:
            nwp_history_minutes = history_minutes
        if nwp_forecast_minutes is None:
            nwp_forecast_minutes = forecast_minutes
        nwp_sequence_len = nwp_history_minutes // 60 + nwp_forecast_minutes // 60 + 1

        self.nwp_embed = nn.Embedding(
            num_embeddings=318,
            embedding_dim=nwp_image_size_pixels**2,
        )

        self.interpolate = nn.Sequential(
            nn.Linear(
                in_features=nwp_sequence_len,
                out_features=self.forecast_len,
            ),
            nn.LeakyReLU(negative_slope=0.01),
        )

        with torch.no_grad():
            # Initate the embedding to be all ones and thus take a simple mean
            self.nwp_embed.weight.copy_(torch.ones(self.nwp_embed.weight.shape))
            # Initiate the linear layer to take a mean across all time steps for each output
            self.interpolate[0].weight.copy_(
                torch.ones(self.interpolate[0].weight.shape) / nwp_sequence_len
            )
            self.interpolate[0].bias.copy_(torch.zeros(self.interpolate[0].bias.shape))

        self.save_hyperparameters()

    def forward(self, x):
        """Run model forward"""
        nwp_data = x[BatchKey.nwp].float()

        # This hack is specific to the current pvnet pipeline. In the pipeline, the dwsrf is
        # standardised, so has mean zero and some negative values. I want all values to be >=0 for
        # this model, so we can calculate a weighted mean for each time step.
        dwsrf = nwp_data[:, :, self.dwsrf_channel]
        mn = 111.28265039
        std = 190.47216887
        dwsrf = dwsrf + (mn / std)

        id = x[BatchKey.gsp_id][:, 0].int()

        mask = self.nwp_embed(id)
        mask = mask.reshape((-1, 1, *dwsrf.shape[-2:]))

        weighted_dwsrf = (mask * dwsrf).mean(dim=-1).mean(dim=-1)

        out = self.interpolate(weighted_dwsrf)

        return out
