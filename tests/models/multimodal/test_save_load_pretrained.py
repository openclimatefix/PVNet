from pvnet.models.base_model import BaseModel
from pathlib import Path


def test_from_pretrained():
    model_name = "openclimatefix/pvnet_uk_region"
    model_version = "92266cd9040c590a9e90ee33eafd0e7b92548be8"

    _ = BaseModel.from_pretrained(
        model_id=model_name,
        revision=model_version,
    )


def test_save_pretrained(tmp_path, multimodal_model, raw_multimodal_model_kwargs):
    data_config_path = "tests/test_data/presaved_samples_uk_regional/data_configuration.yaml"

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
