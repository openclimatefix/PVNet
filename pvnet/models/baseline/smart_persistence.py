"""
Module for the Smart Persistence model.

This module implements a baseline model using smart persistence 
for forecasting global solar power (GSP) yield.
"""

import pvnet
from pvnet.models.base_model import BaseModel
from pvnet.optimizers import AbstractOptimizer
import pvlib
import pandas as pd
from datetime import timedelta
import torch


class Model(BaseModel):
    """A smart persistence model for solar power forecasting."""

    name = "smart_persistence"

    def __init__(
        self,
        forecast_minutes: int = 12,
        history_minutes: int = 6,
        latitude: float = 52.0,
        longitude: float = 0.1,
        tz: str = "Etc/UTC",
        altitude: float = 0.0,
        optimizer: AbstractOptimizer = pvnet.optimizers.Adam(),
    ):
        """
        Initialize the SmartPersistence model.

        Args:
            forecast_minutes (int): Number of minutes to forecast ahead.
            history_minutes (int): Number of minutes of historical data used.
            latitude (float): Latitude of the location.
            longitude (float): Longitude of the location.
            tz (str): Timezone of the location.
            altitude (float): Altitude of the location in meters.
            optimizer (AbstractOptimizer): Optimizer instance for training.
        """
        super().__init__(history_minutes, forecast_minutes, optimizer)
        self.latitude = latitude
        self.longitude = longitude
        self.tz = tz
        self.altitude = altitude
        self.location = pvlib.location.Location(latitude, longitude, tz, altitude)
        self.save_hyperparameters()

    def forward(self, x: dict):
        """
        Perform a forward pass to generate a forecast.

        Args:
            x (dict): Input dictionary containing:
                - "gsp": Tensor of past GSP yields.
                - "time": List of timestamps corresponding to the input data.

        Returns:
            torch.Tensor: Forecasted GSP yield tensor.
        """
        gsp_yield = x["gsp"]
        gsp_yield = gsp_yield[..., 0]
        y_hat = gsp_yield[:, -self.forecast_len - 1]
        times = x["time"]
        current_time = times[-self.forecast_len - 1]
        current_time_index = pd.DatetimeIndex([current_time])

        
        cs_current = self.location.get_clearsky(current_time_index, model='ineichen')['ghi'].iloc[0]
        forecast_times = pd.DatetimeIndex(
            [current_time + timedelta(minutes=i) for i in range(1, self.forecast_len + 1)]
        )
        cs_forecast = self.location.get_clearsky(forecast_times, model='ineichen')['ghi']
        if cs_current > 0:
            k = y_hat / cs_current
        else:
            k = torch.zeros_like(y_hat)

        cs_forecast_tensor = torch.tensor(cs_forecast.values, dtype=y_hat.dtype, device=y_hat.device)
        forecast = k.unsqueeze(1) * cs_forecast_tensor.unsqueeze(0)

        return forecast
