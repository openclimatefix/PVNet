"""Utils"""
import logging

import matplotlib.pyplot as plt
import pandas as pd
import pylab
import rich.syntax
import rich.tree
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


PYTORCH_WEIGHTS_NAME = "model_weights.safetensors"
MODEL_CONFIG_NAME = "model_config.yaml"
DATA_CONFIG_NAME = "data_config.yaml"
DATAMODULE_CONFIG_NAME = "datamodule_config.yaml"
FULL_CONFIG_NAME =  "full_experiment_config.yaml"
MODEL_CARD_NAME = "README.md"



def run_config_utilities(config: DictConfig) -> None:
    """A couple of optional utilities.

    Controlled by main config file:
    - forcing debug friendly configuration

    Modifies DictConfig in place.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    # Enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # Force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        logger.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # Disable adding new keys to config
    OmegaConf.set_struct(config, True)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: tuple[str] = (
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


def plot_batch_forecasts(
    batch,
    y_hat,
    batch_idx=None,
    quantiles=None,
    key_to_plot: str = "gsp",
):
    """Plot a batch of data and the forecast from that batch"""

    y_key = key_to_plot
    y_id_key = f"{key_to_plot}_id"
    time_utc_key = f"{key_to_plot}_time_utc"
    y = batch[y_key].cpu().numpy()  # Select the one it is trained on
    y_hat = y_hat.cpu().numpy()
    # Select between the timesteps in timesteps to plot
    plotting_name = key_to_plot.upper()

    gsp_ids = batch[y_id_key].cpu().numpy().squeeze()

    times_utc = batch[time_utc_key].cpu().numpy().squeeze().astype("datetime64[ns]")
    times_utc = [pd.to_datetime(t) for t in times_utc]

    batch_size = y.shape[0]

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

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
        title = f"Normed {plotting_name} output : batch_idx={batch_idx}"
    else:
        title = f"Normed {plotting_name} output"
    plt.suptitle(title)
    plt.tight_layout()

    return fig
