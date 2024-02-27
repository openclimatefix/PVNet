import os
import glob
import tempfile
import yaml

import hydra
import pytest

from torch.optim import SGD
import torch

import pvnet
from pvnet.models.multimodal.unimodal_teacher import Model


@pytest.fixture
def teacher_dir(multimodal_model, raw_multimodal_model_kwargs):
    
    raw_multimodal_model_kwargs["_target_"] = "pvnet.models.multimodal.multimodal.Model"
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        
        # Save teachers for these modes
        for mode in ["sat", "nwp_ukv"]:
        
            mode_dir = f"{tmpdirname}/{mode}"
            os.mkdir(mode_dir)

            # Checkpoint paths would be like: epoch={X}-step={N}.ckpt or last.ckpt
            path = os.path.join(mode_dir, "epoch=2-step=35002.ckpt")
            path = f"{mode_dir}/epoch=2-step=35002.ckpt"
            
            # Save out themodel config file
            with open(os.path.join(mode_dir, "model_config.yaml"), 'w') as outfile:
                yaml.dump(raw_multimodal_model_kwargs, outfile)
            
            # Save the weights
            torch.save({'model_state_dict': multimodal_model.state_dict()}, path)
        
        yield tempfile
    

@pytest.fixture   
def unimodal_model_kwargs(teacher_dir, model_minutes_kwargs):
    
    # Configure the fusion network
    kwargs = dict(
        output_network=dict(
            _target_=pvnet.models.multimodal.linear_networks.networks.ResFCNet2,
            _partial_=True,
            fc_hidden_features=128,
            n_res_blocks=6,
            res_block_layers=2,
            dropout_frac=0.0,
        ),
    )
    
    # Get the teacher model save directories
    mode_dirs = glob.glob(f"{teacher_dir}/*")
    mode_teacher_dict = dict()
    for mode_dir in mode_dirs:
        mode_name = mode_dir.split("/")[-1].replace("nwp_", "nwp/")
        mode_teacher_dict[mode_name] = mode_dir
    kwargs["mode_teacher_dict"] = mode_teacher_dict
    
    # Add the forecast and history minutes to be compatible with the sample batch
    kwargs.update(model_minutes_kwargs)
    
    yield hydra.utils.instantiate(kwargs)
        
    

@pytest.fixture   
def unimodal_teacher_model(unimodal_model_kwargs):
    return Model(**unimodal_model_kwargs)


def test_model_init(unimodal_model_kwargs):
    Model(**unimodal_model_kwargs)


def test_model_forward(unimodal_teacher_model, sample_batch):
    y = unimodal_teacher_model(sample_batch)

    # check output is the correct shape
    # batch size=2, forecast_len=15
    assert tuple(y.shape) == (2, 16), y.shape


def test_model_backward(unimodal_teacher_model, sample_batch):
    opt = SGD(unimodal_teacher_model.parameters(), lr=0.001)

    y = unimodal_teacher_model(sample_batch)

    # Backwards on sum drives sum to zero
    y.sum().backward()
