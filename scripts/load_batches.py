"""
A test script to examine the pre-saved batches
"""

import torch
from torchdata.datapipes.iter import FileLister

from pvnet.data.datamodule import BatchSplitter


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
