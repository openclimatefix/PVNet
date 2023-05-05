"""
A test script to examine the pre-saved batches
"""

import torch
from ocf_datapipes.utils.consts import BatchKey
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import FileLister, IterDataPipe


def split_batches(batch):
    n_samples = batch[BatchKey.gsp].shape[0]
    keys = list(batch.keys())
    examples = [{} for _ in range(n_samples)]
    for i in range(n_samples):
        b = examples[i]
        for k in keys:
            if ("idx" in k.name) or ("channel_names" in k.name):
                b[k] = batch[k]
            else:
                b[k] = batch[k][i]
    return examples


@functional_datapipe("split_batches")
class BatchSplitter(IterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe):
        """ """
        self.source_datapipe = source_datapipe

    def __iter__(self):
        """Opens the NWP data"""
        for batch in self.source_datapipe:
            for example in split_batches(batch):
                yield example


def get_batch_datapipe(folder, rebatch=False):

    dp = FileLister(root=folder, masks="*.pt", recursive=False)
    if rebatch:
        dp = dp.shuffle(buffer_size=100).sharding_filter()
    dp = dp.map(torch.load)
    if rebatch:
        # Reshuffles the batches loaded from disk
        dp = BatchSplitter(dp)
        dp = dp.shuffle(buffer_size=8 * 16)
        dp = dp.batch(8)
    return dp


if __name__ == "__main__":

    dp = get_batch_datapipe("../tests/data/sample_batches/train")
    batch_0 = next(iter(dp))
