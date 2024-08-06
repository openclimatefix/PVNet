"""Command line tool to push locally save model checkpoints to huggingface

use:
python checkpoint_to_huggingface.py "path/to/model/checkpoints" \
    --local-path="~/tmp/this_model" \
    --no-push-to-hub
"""

import tempfile

import typer
import wandb

from pvnet.load_model import get_model_from_checkpoints

wandb_repo = "openclimatefix/pvnet2.1"
huggingface_repo = "openclimatefix/pvnet_uk_region"


def push_to_huggingface(
    checkpoint_dir_paths: list[str],
    val_best: bool = True,
    wandb_ids: list[str] | None = [],
    local_path: str | None = None,
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
        wandb_ids=wandb_ids,
        push_to_hub=push_to_hub,
        repo_id=huggingface_repo if push_to_hub else None,
    )

    if local_path is None:
        temp_dir.cleanup()


if __name__ == "__main__":
    typer.run(push_to_huggingface)
