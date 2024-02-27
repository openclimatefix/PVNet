"""Utility functions"""

import logging
import math
from typing import Optional

import numpy as np
import torch
from ocf_datapipes.batch import BatchKey

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class PredAccumulator:
    """A class for accumulating y-predictions using grad accumulation and small batch size.

    Attributes:
        _y_hats (list[torch.Tensor]): List of prediction tensors
    """

    def __init__(self):
        """Prediction accumulator"""
        self._y_hats = []

    def __bool__(self):
        return len(self._y_hats) > 0

    def append(self, y_hat: torch.Tensor):
        """Append a sub-batch of predictions"""
        self._y_hats.append(y_hat)

    def flush(self) -> torch.Tensor:
        """Return all appended predictions as single tensor and remove from accumulated store."""
        y_hat = torch.cat(self._y_hats, dim=0)
        self._y_hats = []
        return y_hat


class DictListAccumulator:
    """Abstract class for accumulating dictionaries of lists"""

    @staticmethod
    def _dict_list_append(d1, d2):
        for k, v in d2.items():
            d1[k].append(v)

    @staticmethod
    def _dict_init_list(d):
        return {k: [v] for k, v in d.items()}


class MetricAccumulator(DictListAccumulator):
    """Dictionary of metrics accumulator.

    A class for accumulating, and finding the mean of logging metrics when using grad
    accumulation and the batch size is small.

    Attributes:
        _metrics (Dict[str, list[float]]): Dictionary containing lists of metrics.
    """

    def __init__(self):
        """Dictionary of metrics accumulator."""
        self._metrics = {}

    def __bool__(self):
        return self._metrics != {}

    def append(self, loss_dict: dict[str, float]):
        """Append lictionary of metrics to self"""
        if not self:
            self._metrics = self._dict_init_list(loss_dict)
        else:
            self._dict_list_append(self._metrics, loss_dict)

    def flush(self) -> dict[str, float]:
        """Calculate mean of all accumulated metrics and clear"""
        mean_metrics = {k: np.mean(v) for k, v in self._metrics.items()}
        self._metrics = {}
        return mean_metrics


class BatchAccumulator(DictListAccumulator):
    """A class for accumulating batches when using grad accumulation and the batch size is small.

    Attributes:
        _batches (Dict[BatchKey, list[torch.Tensor]]): Dictionary containing lists of metrics.
    """

    def __init__(self, key_to_keep: str = "gsp"):
        """Batch accumulator"""
        self._batches = {}
        self.key_to_keep = key_to_keep

    def __bool__(self):
        return self._batches != {}

    # @staticmethod
    def _filter_batch_dict(self, d):
        keep_keys = [
            BatchKey[self.key_to_keep],
            BatchKey[f"{self.key_to_keep}_id"],
            BatchKey[f"{self.key_to_keep}_t0_idx"],
            BatchKey[f"{self.key_to_keep}_time_utc"],
        ]
        return {k: v for k, v in d.items() if k in keep_keys}

    def append(self, batch: dict[BatchKey, list[torch.Tensor]]):
        """Append batch to self"""
        if not self:
            self._batches = self._dict_init_list(self._filter_batch_dict(batch))
        else:
            self._dict_list_append(self._batches, self._filter_batch_dict(batch))

    def flush(self) -> dict[BatchKey, list[torch.Tensor]]:
        """Concatenate all accumulated batches, return, and clear self"""
        batch = {}
        for k, v in self._batches.items():
            if k == BatchKey[f"{self.key_to_keep}_t0_idx"]:
                batch[k] = v[0]
            else:
                batch[k] = torch.cat(v, dim=0)
        self._batches = {}
        return batch


class WeightedLosses:
    """Class: Weighted loss depending on the forecast horizon."""

    def __init__(self, decay_rate: Optional[int] = None, forecast_length: int = 6):
        """
        Want to set up the MSE loss function so the weights only have to be calculated once.

        Args:
            decay_rate: The weights exponentially decay depending on the 'decay_rate'.
            forecast_length: The forecast length is needed to make sure the weights sum to 1
        """
        self.decay_rate = decay_rate
        self.forecast_length = forecast_length

        logger.debug(
            f"Setting up weights with decay rate {decay_rate} and of length {forecast_length}"
        )

        # set default rate of ln(2) if not set
        if self.decay_rate is None:
            self.decay_rate = math.log(2)

        # make weights from decay rate
        weights = torch.from_numpy(np.exp(-self.decay_rate * np.arange(self.forecast_length)))

        # normalized the weights, so there mean is 1.
        # To calculate the loss, we times the weights by the differences between truth
        # and predictions and then take the mean across all forecast horizons and the batch
        self.weights = weights / weights.mean()

    def get_mse_exp(self, output, target):
        """Loss function weighted MSE"""

        weights = self.weights.to(target.device)
        # get the differences weighted by the forecast horizon weights
        diff_with_weights = weights * ((output - target) ** 2)

        # average across batches
        return torch.mean(diff_with_weights)

    def get_mae_exp(self, output, target):
        """Loss function weighted MAE"""

        weights = self.weights.to(target.device)
        # get the differences weighted by the forecast horizon weights
        diff_with_weights = weights * torch.abs(output - target)

        # average across batches
        return torch.mean(diff_with_weights)
