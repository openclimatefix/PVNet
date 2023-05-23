"""Training"""

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
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from pvnet import utils

log = utils.get_logger(__name__)

torch.set_default_dtype(torch.float32)


def _callbacks_to_phase(callbacks, phase):
    for c in callbacks:
        if hasattr(c, "switch_phase"):
            c.switch_phase(phase)


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
    logger: list[Logger] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning callbacks
    callbacks: list[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    should_pretrain = False
    for c in callbacks:
        should_pretrain |= hasattr(c, "training_phase") and c.training_phase == "pretrain"

    if should_pretrain:
        _callbacks_to_phase(callbacks, "pretrain")

    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=logger,
        _convert_="partial",
        callbacks=callbacks,
    )

    if should_pretrain:
        # Pre-train the model
        datamodule.block_nwp_and_sat = True
        trainer.fit(model=model, datamodule=datamodule)

    _callbacks_to_phase(callbacks, "main")

    datamodule.block_nwp_and_sat = False
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
        logger=logger,
    )

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]
