"""Command line tool to push locally save model checkpoints to huggingface

use:
python checkpoint_to_huggingface.py "path/to/model/checkpoints" \
    --huggingface-repo="openclimatefix/pvnet_uk_region" \
    --wandb-repo="openclimatefix/pvnet2.1" \
    --local-path="~/tmp/this_model" \
    --no-push-to-hub
"""

import tempfile

import typer
import wandb

from pvnet.load_model import get_model_from_checkpoints


def push_to_huggingface(
    checkpoint_dir_paths: list[str],
    huggingface_repo: str = "openclimatefix/pvnet_uk_region",  # e.g. openclimatefix/windnet_india
    wandb_repo: str = "openclimatefix/pvnet2.1",
    val_best: bool = True,
    wandb_ids: list[str] = [],
    local_path: str = None,
    push_to_hub: bool = True,
    revision: str = "main",
):
    """Push a local model to a huggingface model repo

    Args:
        checkpoint_dir_paths: Path(s) of the checkpoint directory(ies)
        huggingface_repo: Name of the HuggingFace repo to push the model to
        wandb_repo: Name of the wandb repo which has training logs
        val_best: Use best model according to val loss, else last saved model
        wandb_ids: The wandb ID code(s)
        local_path: Where to save the local copy of the model
        push_to_hub: Whether to push the model to the hub or just create local version.
        revision: The git revision to commit from. Only used if push_to_hub is True.
    """

    assert push_to_hub or local_path is not None

    is_ensemble = len(checkpoint_dir_paths) > 1

    # Check if checkpoint dir name is wandb run ID
    if wandb_ids == []:
        all_wandb_ids = [run.id for run in wandb.Api().runs(path=wandb_repo)]
        for path in checkpoint_dir_paths:
            dirname = path.split("/")[-1]
            if dirname in all_wandb_ids:
                wandb_ids.append(dirname)
            else:
                wandb_ids.append(None)

    model, model_config, data_config = get_model_from_checkpoints(checkpoint_dir_paths, val_best)

    if not is_ensemble:
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
        wandb_repo=wandb_repo,
        wandb_ids=wandb_ids,
        push_to_hub=push_to_hub,
        repo_id=huggingface_repo if push_to_hub else None,
        revision=revision,
    )

    if local_path is None:
        temp_dir.cleanup()


if __name__ == "__main__":
    typer.run(push_to_huggingface)
