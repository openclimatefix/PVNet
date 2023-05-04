"""
Constructs batches and saves them to disk.

Currently a slightly hacky implementation due to the way the configs are done. This script will use
the same config file currently set to train the model.

use:
```
python save_batches.py +batch_output_dir="/mnt/disks/batches/batches_v0" +num_train_batches=10_000 +num_val_batches=2_000
```

"""
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from pvnet.data.datamodule import batch_to_tensor
from ocf_datapipes.training.pvnet import pvnet_datapipe
from ocf_datapipes.utils.utils import stack_np_examples_into_batch

import os
import shutil
import logging
import sys

from tqdm import tqdm

from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from torchdata.datapipes.iter import IterableWrapper

# Tired of seeing these warnings
import warnings
from sqlalchemy import exc as sa_exc
warnings.filterwarnings("ignore", category=sa_exc.SAWarning)

logger = logging.getLogger(__name__)

logging.basicConfig(stream=sys.stdout, level=logging.ERROR)


class save_batch_func_factory:
    def __init__(self, batch_dir):
        self.batch_dir = batch_dir
        
    def __call__(self, input):
        i, batch = input
        torch.save(batch, f"{self.batch_dir}/{i:06}.pt")
        
        
def get_datapipe(config_path, start_time, end_time, batch_size):
    data_pipeline = pvnet_datapipe(
        config_path, 
        start_time=start_time,
        end_time=end_time,
        experimental=False,
    )

    data_pipeline = (
        data_pipeline
            .batch(batch_size)
            .map(stack_np_examples_into_batch)
            .map(batch_to_tensor)
    )   
    return data_pipeline

def save_batches_with_dataloader(batch_pipe, batch_dir, num_batches, rs_config):
    
    save_func = save_batch_func_factory(batch_dir)
    filenumber_pipe = IterableWrapper(range(num_batches)).sharding_filter()
    save_pipe = filenumber_pipe.zip(batch_pipe).map(save_func)
    
    rs = MultiProcessingReadingService(**rs_config)
    dataloader = DataLoader2(save_pipe, reading_service=rs)
    
    pbar = tqdm(total=num_batches)
    for i, batch in zip(range(num_batches), dataloader):
        pbar.update()
    pbar.close()
    del dataloader

@hydra.main(config_path="../configs/", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig):
    
    config_dm = config.datamodule
    
    # Set up directory
    os.makedirs(config.batch_output_dir, exist_ok=False)
    
    with open(f"{config.batch_output_dir}/datamodule.yaml", 'w') as f:
        f.write(OmegaConf.to_yaml(config.datamodule))
       
    shutil.copyfile(config_dm.configuration, f"{config.batch_output_dir}/data_configuration.yaml")
    
    os.mkdir(f"{config.batch_output_dir}/train")
    os.mkdir(f"{config.batch_output_dir}/val")

    readingservice_config = dict(
        num_workers=config_dm.num_workers,
        multiprocessing_context="spawn",
        worker_prefetch_cnt=config_dm.prefetch_factor,
    )
    
    print("----- Saving val batches -----")
    
    val_batch_pipe = get_datapipe(
        config_dm.configuration, 
        *config_dm.val_period, 
        config_dm.batch_size,
    )
    
    save_batches_with_dataloader(
        batch_pipe=val_batch_pipe,
        batch_dir=f"{config.batch_output_dir}/val",
        num_batches=config.num_val_batches,
        rs_config=readingservice_config
    )
    
    print("----- Saving train batches -----")  
        
    train_batch_pipe = get_datapipe(
        config_dm.configuration, 
        *config_dm.train_period, 
        config_dm.batch_size,
    )

    save_batches_with_dataloader(
        batch_pipe=train_batch_pipe,
        batch_dir=f"{config.batch_output_dir}/train",
        num_batches=config.num_train_batches,
        rs_config=readingservice_config
    )
    
    print("done")
        
if __name__=="__main__":
    main()