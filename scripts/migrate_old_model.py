"""Script to migrate old trained PVNet models (v4.1) which are hosted on huggingface to current
version.
"""
import datetime
import os

import pkg_resources
import yaml
from huggingface_hub import HfApi

from pvnet.models.base_model import BaseModel
from pvnet.utils import MODEL_CARD_NAME, MODEL_CONFIG_NAME

# ------------------------------------------
# USER SETTINGS

# The huggingface commit of the model you want to update
repo_id = "openclimatefix/pvnet_uk_region"
revision = "6feaa986a6bed3cc6c7961c6bf9e92fb15acca6a"

# The local directory which will be downloaded to
local_dir = "/home/jamesfulton/tmp/model_migration"

# Whether to upload the migrated model back to the huggingface repo
upload = True

# ------------------------------------------
# SETUP

os.makedirs(local_dir, exist_ok=False)

# Set up huggingface API
api = HfApi()

# Download the model repo
local_dir = api.snapshot_download(
    repo_id=repo_id,
    revision=revision,
    local_dir=local_dir,
    force_download=True,
)

# ------------------------------------------
# MIGRATIUON STEPS

# Modify the model config
with open(f"{local_dir}/{MODEL_CONFIG_NAME}") as cfg:
    model_config = yaml.load(cfg, Loader=yaml.FullLoader)

del model_config["optimizer"]

with open(f"{local_dir}/{MODEL_CONFIG_NAME}", "w") as f:
    yaml.dump(model_config, f, sort_keys=False, default_flow_style=False)

# Add a note to the model card to say the model has been migrated
with open(f"{local_dir}/{MODEL_CARD_NAME}", "a") as f:
    current_date = datetime.date.today().strftime("%Y-%m-%d")
    pvnet_version = pkg_resources.get_distribution("pvnet").version
    f.write(
        f"\n\n---\n**Migration Note**: This model was migrated on {current_date} "
        f"to pvnet version {pvnet_version}\n"
    )

# ------------------------------------------
# CHECKS

# Check the model can be loaded
model = BaseModel.from_pretrained(model_id=local_dir, revision=None)

print("Model checkpoint successfully migrated")

# ------------------------------------------
# UPLOAD

if upload:
    print("Uploading migrated model to huggingface")

    # Upload back to huggingface
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Migrate model to pvnet version {pvnet_version}",
    )

    # Print the most recent commit hash
    c = api.list_repo_commits(repo_id=repo_id, repo_type="model")[0]

    print(
        f"\nThe latest commit is now: \n"
        f"    date: {c.created_at} \n"
        f"    commit hash: {c.commit_id}\n"
        f"    by: {c.authors}\n"
        f"    title: {c.title}\n"
    )
