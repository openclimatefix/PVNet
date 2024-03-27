"""Utils common between Wind and PV datamodules"""
from ocf_datapipes.batch import BatchKey, unstack_np_batch_into_examples
from torch.utils.data import IterDataPipe, functional_datapipe


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
