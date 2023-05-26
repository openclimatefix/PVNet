import hydra
from pyaml_env import parse_config
import os
import tempfile
import wandb
import typer

def push_to_huggingface(checkpoint_dir_path, val_best=True, wandb_id=None):
    """Push a local model to pvnet_v2 huggingface model repo
    
    checkpoint_dir_path (str): Path of the chekpoint directory
    val_best (bool): Use best model according to val loss, else last saved model
    wandb_id (str): The wandb ID code
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if checkpoint dir name is wandb run ID
    if wandb_id is None:
        all_wandb_ids = [run.id  for run in wandb.Api().runs(path="openclimatefix/pvnet2.1")]
        dirname = checkpoint_dir_path.split("/")[-1]
        if dirname in all_wandb_runs:
            wandb_id = dirname
    
    # Load the model
    model_config = parse_config(f"{checkpoint_dir_path}/model_config.yaml")

    model = hydra.utils.instantiate(model_config)
    checkpoint = torch.load(f"{checkpoint_dir_path}/last.ckpt")

    if val_best:
        # Only one epoch (best) saved per model
        files = glob.glob(f"{checkpoint_dir_path}/epoch*.ckpt")
        assert len(files)==1
        checkpoint = torch.load(files[0])
    else:
        checkpoint = torch.load(f"{checkpoint_dir_path}/last.ckpt")
    
    model.load_state_dict(state_dict=checkpoint['state_dict'])
    
    # Push to hub
    with tempfile.TemporaryDirectory() as tmpdirname:
        model.save_pretrained(
            tmpdirname, 
            config=model_config,
            wandb_model_code=wandb_id,
            push_to_hub=True, 
            repo_id="openclimatefix/pvnet_v2"
        )


if __name__ == "__main__":
    typer.run(push_to_huggingface)