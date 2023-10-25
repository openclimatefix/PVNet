"""Utils"""
import logging
import os
import warnings
from collections.abc import Sequence

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab
import rich.syntax
import rich.tree
import xarray as xr
import yaml
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only
from ocf_datapipes.utils.consts import BatchKey, Location
from ocf_datapipes.utils.geospatial import osgb_to_lon_lat
from omegaconf import DictConfig, OmegaConf

import pvnet


def load_config(config_file):
    """
    Open yam configruation file, and get rid eof '_target_' line
    """

    # get full path of config file
    path = os.path.dirname(pvnet.__file__)
    config_file = f"{path}/../{config_file}"

    with open(config_file) as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)

    if "_target_" in config.keys():
        config.pop("_target_")  # This is only for Hydra

    return config


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


class GSPLocationLookup:
    """Query object for GSP location from GSP ID"""

    def __init__(self, x_osgb: xr.DataArray, y_osgb: xr.DataArray):
        """Query object for GSP location from GSP ID

        Args:
            x_osgb: DataArray of the OSGB x-coordinate for any given GSP ID
            y_osgb: DataArray of the OSGB y-coordinate for any given GSP ID

        """
        self.x_osgb = x_osgb
        self.y_osgb = y_osgb

    def __call__(self, gsp_id: int) -> Location:
        """Returns the locations for the input GSP IDs.

        Args:
            gsp_id: Integer ID of the GSP
        """
        return Location(
            x=self.x_osgb.sel(gsp_id=gsp_id).item(),
            y=self.y_osgb.sel(gsp_id=gsp_id).item(),
            id=gsp_id,
        )


def extras(config: DictConfig) -> None:
    """A couple of optional utilities.

    Controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration

    Modifies DictConfig in place.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger()

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # disable adding new keys to config
    OmegaConf.set_struct(config, True)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "datamodule",
        "callbacks",
        "logger",
        "seed",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)


def empty(*args, **kwargs):
    """Returns nothing"""
    pass


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: list[pl.Callback],
    logger: list[Logger],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - number of trainable model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]
    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = empty


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: list[pl.Callback],
    loggers: list[Logger],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for logger in loggers:
        if isinstance(logger, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()


def plot_batch_forecasts(batch, y_hat, batch_idx=None, quantiles=None):
    """Plot a batch of data and the forecast from that batch"""

    def _get_numpy(key):
        return batch[key].cpu().numpy().squeeze()

    y = batch[BatchKey.gsp].cpu().numpy()
    y_hat = y_hat.cpu().numpy()

    gsp_ids = batch[BatchKey.gsp_id].cpu().numpy().squeeze()
    t0_idx = batch[BatchKey.gsp_t0_idx]

    times_utc = batch[BatchKey.gsp_time_utc].cpu().numpy().squeeze().astype("datetime64[s]")
    times_utc = [pd.to_datetime(t) for t in times_utc]

    len(times_utc[0]) - t0_idx - 1
    batch_size = y.shape[0]

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))

    for i, ax in enumerate(axes.ravel()):
        if i >= batch_size:
            ax.axis("off")
            continue
        ax.plot(times_utc[i], y[i], marker=".", color="k", label=r"$y$")

        if quantiles is None:
            ax.plot(
                times_utc[i][-len(y_hat[i]) :], y_hat[i], marker=".", color="r", label=r"$\hat{y}$"
            )
        else:
            cm = pylab.get_cmap("twilight")
            for nq, q in enumerate(quantiles):
                ax.plot(
                    times_utc[i][-len(y_hat[i]) :],
                    y_hat[i, :, nq],
                    color=cm(q),
                    label=r"$\hat{y}$" + f"({q})",
                    alpha=0.7,
                )

        ax.set_title(f"ID: {gsp_ids[i]} | {times_utc[i][0].date()}", fontsize="small")

        xticks = [t for t in times_utc[i] if t.minute == 0][::2]
        ax.set_xticks(ticks=xticks, labels=[f"{t.hour:02}" for t in xticks], rotation=90)
        ax.grid()

    axes[0, 0].legend(loc="best")

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (hour of day)")

    if batch_idx is not None:
        title = f"Normed GSP output : batch_idx={batch_idx}"
    else:
        title = "Normed GSP output"
    plt.suptitle(title)
    plt.tight_layout()

    return fig


def construct_ocf_ml_metrics_batch_df(batch, y, y_hat):
    """Helper function tot construct DataFrame for ocf_ml_metrics"""

    def _repeat(x):
        return np.repeat(x.squeeze(), n_times)

    def _get_numpy(key):
        return batch[key].cpu().numpy().squeeze()

    t0_idx = batch[BatchKey.gsp_t0_idx]
    times_utc = _get_numpy(BatchKey.gsp_time_utc)
    n_times = len(times_utc[0]) - t0_idx - 1

    y_osgb_centre = _get_numpy(BatchKey.gsp_y_osgb)
    x_osgb_centre = _get_numpy(BatchKey.gsp_x_osgb)
    longitude, latitude = osgb_to_lon_lat(x=x_osgb_centre, y=y_osgb_centre)

    # Store df columns in dict
    df_dict = {}

    # Repeat these features for each forecast time
    df_dict["latitude"] = _repeat(latitude)
    df_dict["longitude"] = _repeat(longitude)
    df_dict["id"] = _repeat(_get_numpy(BatchKey.gsp_id))
    df_dict["t0_datetime_utc"] = _repeat(times_utc[:, t0_idx]).astype("datetime64[s]")
    df_dict["capacity_mwp"] = _repeat(_get_numpy(BatchKey.gsp_capacity_megawatt_power))

    # TODO: Some (10%) of these values are NaN -> 0 for time t0 for pvnet pipeline
    #       Better to search for last non-nan (non-zero)?
    df_dict["t0_actual_pv_outturn_mw"] = _repeat(
        (_get_numpy(BatchKey.gsp_capacity_megawatt_power)[:, None] * _get_numpy(BatchKey.gsp))[
            :, t0_idx
        ]
    )

    # Flatten the forecasts times to 1D
    df_dict["target_datetime_utc"] = times_utc[:, t0_idx + 1 :].flatten().astype("datetime64[s]")
    df_dict["actual_pv_outturn_mw"] = y.cpu().numpy().flatten() * df_dict["capacity_mwp"]
    df_dict["forecast_pv_outturn_mw"] = y_hat.cpu().numpy().flatten() * df_dict["capacity_mwp"]
    return pd.DataFrame(df_dict)
