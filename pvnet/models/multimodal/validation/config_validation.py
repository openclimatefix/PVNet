"""Main validation function for multimodal configuration."""

import logging

from omegaconf import DictConfig, OmegaConf

from pvnet.models.multimodal.validation.config_structure_validator import (
    validate_model_config,
)

logger = logging.getLogger(__name__)


def validate(
    multimodal_config: DictConfig,
) -> None:
    """
    Validate the multimodal model configuration dictionary.

    This function serves as the entry point for static configuration validation.
    It converts the OmegaConf object to a standard dictionary and passes it to the
    structural validator.

    Args:
        multimodal_config: The OmegaConf DictConfig for the multimodal model.

    Raises:
        KeyError, TypeError, ValueError: If static configuration rules are violated.
    """
    logger.info("Starting configuration validation...")

    mm_config_dict: dict = OmegaConf.to_container(
        multimodal_config, resolve=True, throw_on_missing=True
    )

    validate_model_config(mm_config_dict)
    logger.info("Configuration validation completed successfully.")
