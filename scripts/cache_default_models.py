"""A script to download default models from huggingface.

Downloading these model files in the build means we do not need to download them each time the app
is run.
"""

import typer
from pvnet_summation.models.base_model import BaseModel as SummationBaseModel

from pvnet.app import (
    default_model_name,
    default_model_version,
    default_summation_model_name,
    default_summation_model_version,
)
from pvnet.models.base_model import BaseModel as PVNetBaseModel


def main():
    """Download model from Huggingface and save it to cache."""
    # Model will be downloaded and saved to cache on disk
    PVNetBaseModel.from_pretrained(
        default_model_name,
        revision=default_model_version,
    )

    # Model will be downloaded and saved to cache on disk
    SummationBaseModel.from_pretrained(
        default_summation_model_name,
        revision=default_summation_model_version,
    )


if __name__ == "__main__":
    typer.run(main)
