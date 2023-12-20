"""Utils common between Wind and PV datamodules"""
import numpy as np
import torch
from ocf_datapipes.batch import BatchKey, unstack_np_batch_into_examples
from torch.utils.data import IterDataPipe, functional_datapipe


def copy_batch_to_device(batch, device):
    """Moves a dict-batch of tensors to new device."""
    batch_copy = {}
    for k in list(batch.keys()):
        if isinstance(batch[k], torch.Tensor):
            batch_copy[k] = batch[k].to(device)
        else:
            batch_copy[k] = batch[k]
    return batch_copy


def batch_to_tensor(batch):
    """Moves numpy batch to a tensor"""
    for k in list(batch.keys()):
        if isinstance(batch[k], np.ndarray) and np.issubdtype(batch[k].dtype, np.number):
            batch[k] = torch.as_tensor(batch[k])
    return batch


@functional_datapipe("split_batches")
class BatchSplitter(IterDataPipe):
    """Pipeline step to split batches of data and yield single examples"""

    def __init__(self, source_datapipe: IterDataPipe, splitting_key: BatchKey = BatchKey.gsp):
        """Pipeline step to split batches of data and yield single examples"""
        self.source_datapipe = source_datapipe
        self.splitting_key = splitting_key

    def __iter__(self):
        """Opens the NWP data"""
        for batch in self.source_datapipe:
            for example in unstack_np_batch_into_examples(batch):
                yield example
