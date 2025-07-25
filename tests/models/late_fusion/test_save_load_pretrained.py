import pkg_resources
from pvnet.models import BaseModel
import pvnet.model_cards
import yaml
import tempfile


card_path = f"{pvnet.model_cards.__path__[0]}/empty_model_card_template.md"


def test_save_pretrained(tmp_path, late_fusion_model, raw_late_fusion_model_kwargs, sample_datamodule):
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
    model_config = {"_target_": "pvnet.models.LateFusionModel"}
    model_config.update(raw_late_fusion_model_kwargs)

    # Save the model
    model_output_dir = f"{tmp_path}/model"
    late_fusion_model.save_pretrained(
        save_directory=model_output_dir,
        model_config=model_config,
        data_config_path=data_config_path,
        wandb_repo="test",
        wandb_ids="abc",
        card_template_path=card_path,
        push_to_hub=False,
    )

    # Load the model
    _ = BaseModel.from_pretrained(model_id=model_output_dir, revision=None)


def test_create_hugging_face_model_card():

    # Create Hugging Face ModelCard
    card = BaseModel.create_hugging_face_model_card(card_path, wandb_repo="test", wandb_ids="abc")

    # Extract the card markdown
    card_markdown = card.content

    # Regex to find if the pvnet and ocf-data-sampler versions are present
    pvnet_version = pkg_resources.get_distribution("pvnet").version
    has_pvnet = f"pvnet=={pvnet_version}" in card_markdown

    ocf_sampler_version = pkg_resources.get_distribution("ocf-data-sampler").version
    has_ocf_data_sampler= f"ocf-data-sampler=={ocf_sampler_version}" in card_markdown

    assert has_pvnet, f"The hugging face card created does not display the PVNet package version"
    assert has_ocf_data_sampler, f"The hugging face card created does not display the ocf-data-sampler package version"
