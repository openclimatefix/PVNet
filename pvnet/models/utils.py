"""Utility functions"""

import numpy as np
import torch
from ocf_datapipes.utils.consts import BatchKey


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
        self._y_hats += [y_hat]

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
            d1[k] += [v]

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

    def __init__(self):
        """Batch accumulator"""
        self._batches = {}

    def __bool__(self):
        return self._batches != {}

    @staticmethod
    def _filter_batch_dict(d):
        keep_keys = [BatchKey.gsp, BatchKey.gsp_id, BatchKey.gsp_t0_idx, BatchKey.gsp_time_utc]
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
            if k == BatchKey.gsp_t0_idx:
                batch[k] = v[0]
            else:
                batch[k] = torch.cat(v, dim=0)
        self._batches = {}
        return batch
