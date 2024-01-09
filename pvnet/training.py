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


def _callbacks_to_phase(callbacks, phase):
    for c in callbacks:
        if hasattr(c, "switch_phase"):
            c.switch_phase(phase)


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
                    data_config = f"{config.datamodule.batch_dir}/data_configuration.yaml"

                assert os.path.isfile(data_config), f"Data config file not found: {data_config}"
                shutil.copyfile(data_config, f"{callback.dirpath}/data_config.yaml")
                break

    should_pretrain = False
    for c in callbacks:
        should_pretrain |= hasattr(c, "training_phase") and c.training_phase == "pretrain"

    if should_pretrain:
        _callbacks_to_phase(callbacks, "pretrain")

    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=loggers,
        _convert_="partial",
        callbacks=callbacks,
    )

    if should_pretrain:
        # Pre-train the model
        raise NotImplementedError("Pre-training is not yet supported")
        # The parameter `block_nwp_and_sat` has been removed from datapipes
        # If pretraining is re-supported in the future it is likely any pre-training logic should
        # go here or perhaps in the callbacks
        # datamodule.block_nwp_and_sat = True

        trainer.fit(model=model, datamodule=datamodule)

    _callbacks_to_phase(callbacks, "main")

    trainer.should_stop = False

    # Train the model completely
    trainer.fit(model=model, datamodule=datamodule)

    if config.test_after_training:
        # Evaluate model on test set, using the best model achieved during training
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        loggers=loggers,
    )

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]
