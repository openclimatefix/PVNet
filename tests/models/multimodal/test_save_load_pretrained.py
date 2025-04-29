import pytest
import re
from pvnet.models.base_model import BaseModel
from pathlib import Path
import yaml
import tempfile


def test_from_pretrained():
    model_name = "openclimatefix/pvnet_uk_region"
    model_version = "92266cd9040c590a9e90ee33eafd0e7b92548be8"

    _ = BaseModel.from_pretrained(
        model_id=model_name,
        revision=model_version,
    )


def test_save_pretrained(tmp_path, multimodal_model, raw_multimodal_model_kwargs, sample_datamodule):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        # Get sample directory from the datamodule
        sample_dir = sample_datamodule.sample_dir

        # Create config with matching structure
        data_config = {
            "general": {
                "description": "Config for training the saved PVNet model",
                "name": "test_pvnet"
            },
            "input_data": {
                "gsp": {
                    "zarr_path": sample_dir,
                    "interval_start_minutes": -120,
                    "interval_end_minutes": 480,
                    "time_resolution_minutes": 30,
                    "dropout_timedeltas_minutes": None,
                    "dropout_fraction": 0
                },
                "nwp": {
                    "ukv": {
                        "provider": "ukv",
                        "zarr_path": sample_dir,
                        "interval_start_minutes": -120,
                        "interval_end_minutes": 480,
                        "time_resolution_minutes": 60,
                        "channels": ["t", "dswrf", "dlwrf"],
                        "image_size_pixels_height": 24,
                        "image_size_pixels_width": 24,
                        "dropout_timedeltas_minutes": None,
                        "dropout_fraction": 0,
                        "max_staleness_minutes": None
                    }
                },
                "satellite": {
                    "zarr_path": sample_dir,
                    "interval_start_minutes": -30,
                    "interval_end_minutes": 0,
                    "time_resolution_minutes": 5,
                    "channels": ["IR_016", "IR_039", "IR_087"],
                    "image_size_pixels_height": 24,
                    "image_size_pixels_width": 24,
                    "dropout_timedeltas_minutes": None,
                    "dropout_fraction": 0
                }
            },
            "sample_dir": sample_dir,
            "train_period": [None, None],
            "val_period": [None, None],
            "test_period": [None, None]
        }

        yaml.dump(data_config, temp_file)
        data_config_path = temp_file.name

    # Construct the model config
    model_config = {"_target_": "pvnet.models.multimodal.multimodal.Model"}
    model_config.update(raw_multimodal_model_kwargs)

    # Save the model
    model_output_dir = f"{tmp_path}/model"
    multimodal_model.save_pretrained(
        model_output_dir,
        config=model_config,
        data_config=data_config_path,
        wandb_repo=None,
        wandb_ids="excluded-for-text",
        push_to_hub=False,
        repo_id="openclimatefix/pvnet_uk_region",
        revision="main",
    )

    # Load the model
    _ = BaseModel.from_pretrained(
        model_id=model_output_dir,
        revision=None,
    )

@pytest.mark.parametrize(
    "repo_id, wandb_repo, wandb_ids",
    [
        (
            "openclimatefix/pvnet_uk_region",
            "None",
            "excluded-for-text"
        ),
    ],
)
def test_create_hugging_face_model_card(repo_id, wandb_repo, wandb_ids):

    # Create Hugging Face ModelCard
    card = BaseModel.create_hugging_face_model_card(
        repo_id=repo_id,
        wandb_repo=wandb_repo,
        wandb_ids=wandb_ids
    )

    # Extract the card markdown
    card_markdown = card.content

    # Regex to find if the pvnet and ocf-data-sampler versions are present
    has_pvnet = re.search(r'^ - pvnet==', card_markdown, re.IGNORECASE | re.MULTILINE)
    has_ocf_data_sampler= re.search(r'^ - ocf[-_]data[-_]sampler==', card_markdown, re.IGNORECASE | re.MULTILINE)

    assert not has_pvnet, "The hugging face card created does not display the PVNet package version"
    assert has_ocf_data_sampler, "The hugging face card created does not display the ocf-data-sampler package version"
