"""Load a model from its checkpoint directory"""

import glob
import os
from typing import Any

import hydra
import torch
from pyaml_env import parse_config

from pvnet.models.ensemble import Ensemble
from pvnet.utils import (
    DATA_CONFIG_NAME,
    DATAMODULE_CONFIG_NAME,
    MODEL_CONFIG_NAME,
)


def get_model_from_checkpoints(
    checkpoint_dir_paths: list[str],
    val_best: bool = True,
) -> tuple[torch.nn.Module, dict[str, Any], str, str | None]:
    """Load a model from its checkpoint directory

    Returns:
        tuple:
            model: nn.Module of pretrained model.
            model_config: path to model config used to train the model.
            data_config: path to data config used to create samples for the model.
            datamodule_config: path to datamodule used to create samples e.g train/test split info.

    """
    is_ensemble = len(checkpoint_dir_paths) > 1

    model_configs = []
    models = []
    data_configs = []
    datamodule_configs = []

    for path in checkpoint_dir_paths:
        # Load lightning training module
        model_config = parse_config(f"{path}/{MODEL_CONFIG_NAME}")
        lightning_module = hydra.utils.instantiate(model_config)

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

        lightning_module.load_state_dict(state_dict=checkpoint["state_dict"])

        # Extract the model from the lightning module
        models.append(lightning_module.model)
        model_configs.append(model_config["model"])

        # Store the data config used for the model
        data_config = f"{path}/{DATA_CONFIG_NAME}"

        if os.path.isfile(data_config):
            data_configs.append(data_config)
        else:
            raise FileNotFoundError(f"File {data_config} does not exist")

        # Check for datamodule config
        datamodule_config = f"{path}/{DATAMODULE_CONFIG_NAME}"
        if os.path.isfile(datamodule_config):
            datamodule_configs.append(datamodule_config)
        else:
            datamodule_configs.append(None)

    if is_ensemble:
        model_config = {
            "_target_": "pvnet.models.ensemble.Ensemble",
            "model_list": model_configs,
        }
        model = Ensemble(model_list=models)

    else:
        model_config = model_configs[0]
        model = models[0]

    # Assume if using an ensemble that the members were trained on the same input data
    data_config = data_configs[0]
    datamodule_config = datamodule_configs[0]

    return model, model_config, data_config, datamodule_config
