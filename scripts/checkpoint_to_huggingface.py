"""Command line tool to push locally save model checkpoints to huggingface

use:
python checkpoint_to_huggingface.py "path/to/model/checkpoints" \
    --local-path="~/tmp/this_model" \
    --no-push-to-hub
"""

import glob
import os
import tempfile
from typing import Optional

import hydra
import torch
import typer
import wandb
from pyaml_env import parse_config

from pvnet.models.ensemble import Ensemble
from pvnet.models.multimodal.unimodal_teacher import Model as UMTModel


def push_to_huggingface(
    checkpoint_dir_paths: list[str],
    val_best: bool = True,
    wandb_ids: Optional[list[str]] = None,
    local_path: Optional[str] = None,
    push_to_hub: bool = True,
):
    """Push a local model to pvnet_v2 huggingface model repo

    Args:
        checkpoint_dir_paths: Path(s) of the checkpoint directory(ies)
        val_best: Use best model according to val loss, else last saved model
        wandb_ids: The wandb ID code(s)
        local_path: Where to save the local copy of the model
        push_to_hub: Whether to push the model to the hub or just create local version.
    """

    assert push_to_hub or local_path is not None

    os.path.dirname(os.path.abspath(__file__))

    is_ensemble = len(checkpoint_dir_paths) > 1

    # Check if checkpoint dir name is wandb run ID
    if wandb_ids == []:
        all_wandb_ids = [run.id for run in wandb.Api().runs(path="openclimatefix/pvnet2.1")]
        for path in checkpoint_dir_paths:
            dirname = path.split("/")[-1]
            if dirname in all_wandb_ids:
                wandb_ids.append(dirname)
            else:
                wandb_ids.append(None)

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
            assert len(files) == 1
            checkpoint = torch.load(files[0], map_location="cpu")
        else:
            checkpoint = torch.load(f"{path}/last.ckpt", map_location="cpu")

        model.load_state_dict(state_dict=checkpoint["state_dict"])

        if isinstance(model, UMTModel):
            model, model_config = model.convert_to_multimodal_model(model_config)

        # Check for data config
        data_config = f"{path}/data_config.yaml"
        assert os.path.isfile(data_config)

        model_configs.append(model_config)
        models.append(model)
        data_configs.append(data_config)

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
        wandb_ids = wandb_ids[0]

    # Push to hub
    if local_path is None:
        temp_dir = tempfile.TemporaryDirectory()
        model_output_dir = temp_dir.name
    else:
        model_output_dir = local_path

    model.save_pretrained(
        model_output_dir,
        config=model_config,
        data_config=data_config,
        wandb_ids=wandb_ids,
        push_to_hub=push_to_hub,
        repo_id="openclimatefix/pvnet_v2" if push_to_hub else None,
    )

    if local_path is None:
        temp_dir.cleanup()


if __name__ == "__main__":
    typer.run(push_to_huggingface)
