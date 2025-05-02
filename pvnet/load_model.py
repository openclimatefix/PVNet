""" Load a model from its checkpoint directory """
import glob
import os

import hydra
import torch
from pyaml_env import parse_config

from pvnet.models.ensemble import Ensemble
from pvnet.models.multimodal.unimodal_teacher import Model as UMTModel


def get_model_from_checkpoints(
    checkpoint_dir_paths: list[str],
    val_best: bool = True,
):
    """Load a model from its checkpoint directory"""
    is_ensemble = len(checkpoint_dir_paths) > 1

    model_configs = []
    models = []
    data_configs = []

    for path in checkpoint_dir_paths:
        # Load the model
        model_config = parse_config(f"{path}/model_config.yaml")

        model = hydra.utils.instantiate(model_config)

        if val_best:
            # Only one epoch (best) saved per model
            files = glob.glob(f"{path}/epoch*.ckpt")
            if len(files) != 1:
                raise ValueError(
                    f"Found {len(files)} checkpoints @ {path}/epoch*.ckpt. Expected one."
                )
            # TODO: Loading with weights_only=False is not recommended
            checkpoint = torch.load(files[0], map_location="cpu", weights_only=False)
        else:
            checkpoint = torch.load(f"{path}/last.ckpt", map_location="cpu", weights_only=False)

        model.load_state_dict(state_dict=checkpoint["state_dict"])

        if isinstance(model, UMTModel):
            model, model_config = model.convert_to_multimodal_model(model_config)

        # Check for data config
        data_config = f"{path}/data_config.yaml"

        if os.path.isfile(data_config):
            data_configs.append(data_config)
        else:
            data_configs.append(None)

        model_configs.append(model_config)
        models.append(model)

    if is_ensemble:
        model_config = {
            "_target_": "pvnet.models.ensemble.Ensemble",
            "model_list": model_configs,
        }
        model = Ensemble(model_list=models)
        data_config = data_configs[0]

    else:
        model_config = model_configs[0]
        model = models[0]
        data_config = data_configs[0]

    return model, model_config, data_config
