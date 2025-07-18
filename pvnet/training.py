"""Training"""
import os
import shutil
from typing import Optional

import hydra
import torch
from lightning.pytorch import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf

from pvnet import utils

log = utils.get_logger(__name__)

torch.set_default_dtype(torch.float32)


def resolve_monitor_loss(output_quantiles):
    """Return the desired metric to monitor based on whether quantile regression is being used.

    The adds the option to use something like:
        monitor: "${resolve_monitor_loss:${model.output_quantiles}}"

    in early stopping and model checkpoint callbacks so the callbacks config does not need to be
    modified depending on whether quantile regression is being used or not.
    """
    if output_quantiles is None:
        return "MAE/val"
    else:
        return "quantile_loss/val"


OmegaConf.register_new_resolver("resolve_monitor_loss", resolve_monitor_loss)


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.

    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Init lightning loggers
    loggers: list[Logger] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                loggers.append(hydra.utils.instantiate(lg_conf))

    # Init lightning callbacks
    callbacks: list[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Align the wandb id with the checkpoint path
    # - only works if wandb logger and model checkpoint used
    # - this makes it easy to push the model to huggingface
    use_wandb_logger = False
    for logger in loggers:
        log.info(f"{logger}")
        if isinstance(logger, WandbLogger):
            use_wandb_logger = True
            wandb_logger = logger
            break

    if use_wandb_logger:
        for callback in callbacks:
            log.info(f"{callback}")
            if isinstance(callback, ModelCheckpoint):
                # Need to call the .experiment property to initialise the logger
                wandb_logger.experiment
                callback.dirpath = "/".join(
                    callback.dirpath.split("/")[:-1] + [wandb_logger.version]
                )
                # Also save model config here - this makes for easy model push to huggingface
                os.makedirs(callback.dirpath, exist_ok=True)
                OmegaConf.save(config.model, f"{callback.dirpath}/model_config.yaml")

                # Similarly save the data config
                data_config = config.datamodule.configuration
                if data_config is None:
                    # Data config can be none if using presaved batches. We go to the presaved
                    # batches to get the data config
                    data_config = f"{config.datamodule.sample_dir}/data_configuration.yaml"

                    # Also save additonal datamodule config related to presaved batches
                    datamodule_config = f"{config.datamodule.sample_dir}/datamodule.yaml"
                    shutil.copyfile(datamodule_config, f"{callback.dirpath}/datamodule_config.yaml")

                assert os.path.isfile(data_config), f"Data config file not found: {data_config}"
                shutil.copyfile(data_config, f"{callback.dirpath}/data_config.yaml")

                # upload configuration up to wandb
                OmegaConf.save(config, "./experiment_config.yaml")
                wandb_logger.experiment.save(
                    f"{callback.dirpath}/data_config.yaml", callback.dirpath
                )
                wandb_logger.experiment.save("./experiment_config.yaml")

                break

    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=loggers,
        _convert_="partial",
        callbacks=callbacks,
    )

    # Train the model completely
    trainer.fit(model=model, datamodule=datamodule)
