from ocf_datapipes.config.load import load_yaml_configuration
from ocf_datapipes.training.gsp_pv_satellite_nwp import gsp_pv_nwp_satellite_data_pipeline
from ocf_datapipes.training.pv_satellite_nwp import pv_nwp_satellite_data_pipeline
from ocf_datapipes.training.simple_pv import simple_pv_datapipe
from ocf_datapipes.training.nwp_pv import nwp_pv_datapipe

import time

from torch.utils.data import DataLoader2

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, "INFO"),
    format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
)


file = './pvnet/data/gcp_configuration.yaml'

config = load_yaml_configuration(file)
config.process.batch_size = 2


# just load pv data
for pipeline_object in [simple_pv_datapipe,nwp_pv_datapipe, pv_nwp_satellite_data_pipeline, gsp_pv_nwp_satellite_data_pipeline]:
    pipeline = pipeline_object(config).set_length(5)
    dl = DataLoader2(dataset=pipeline, batch_size=None, num_workers=0)
    dl_iter = iter(pipeline)

    t0 = time.time()
    batch = next(dl_iter)
    t1 = time.time()
    print(f'First batch took {t1-t0}s ({pipeline_object})')

    t0 = time.time()
    batch = next(dl_iter)
    t1 = time.time()
    print(f'Second batch took {t1-t0}s ({pipeline_object})')

    del dl
    del pipeline
