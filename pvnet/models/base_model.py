"""Base model for all PVNet submodels"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Union

import hydra
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
import yaml
from huggingface_hub import ModelCard, ModelCardData, PyTorchModelHubMixin
from huggingface_hub.constants import CONFIG_NAME, PYTORCH_WEIGHTS_NAME
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils._deprecation import _deprecate_positional_args
from ocf_datapipes.batch import BatchKey
from ocf_ml_metrics.evaluation.evaluation import evaluation

from pvnet.models.utils import (
    BatchAccumulator,
    MetricAccumulator,
    PredAccumulator,
    WeightedLosses,
)
from pvnet.optimizers import AbstractOptimizer
from pvnet.utils import construct_ocf_ml_metrics_batch_df, plot_batch_forecasts

DATA_CONFIG_NAME = "data_config.yaml"


logger = logging.getLogger(__name__)

activities = [torch.profiler.ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(torch.profiler.ProfilerActivity.CUDA)


def make_clean_data_config(input_path, output_path, placeholder="PLACEHOLDER"):
    """Resave the data config and replace the filepaths with a placeholder.

    Args:
        input_path: Path to input datapipes configuration file
        output_path: Location to save the output configuration file
        placeholder: String placeholder for data sources
    """
    with open(input_path) as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)

    config["general"]["description"] = "Config for training the saved PVNet model"
    config["general"]["name"] = "PVNet current"

    for source in ["gsp", "satellite", "hrvsatellite"]:
        if source in config["input_data"]:
            # If not empty - i.e. if used
            if config["input_data"][source][f"{source}_zarr_path"] != "":
                config["input_data"][source][f"{source}_zarr_path"] = f"{placeholder}.zarr"

    if "nwp" in config["input_data"]:
        for source in config["input_data"]["nwp"]:
            if config["input_data"]["nwp"][source]["nwp_zarr_path"] != "":
                config["input_data"]["nwp"][source]["nwp_zarr_path"] = f"{placeholder}.zarr"

    if "pv" in config["input_data"]:
        for d in config["input_data"]["pv"]["pv_files_groups"]:
            d["pv_filename"] = f"{placeholder}.netcdf"
            d["pv_metadata_filename"] = f"{placeholder}.csv"

    if "sensor" in config["input_data"]:
        # If not empty - i.e. if used
        if config["input_data"][source][f"{source}_filename"] != "":
            config["input_data"][source][f"{source}_filename"] = f"{placeholder}.nc"

    with open(output_path, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


class PVNetModelHubMixin(PyTorchModelHubMixin):
    """
    Implementation of [`PyTorchModelHubMixin`] to provide model Hub upload/download capabilities.
    """

    @classmethod
    @_deprecate_positional_args(version="0.16")
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str,
        cache_dir: str,
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = False,
        **model_kwargs,
    ):
        """Load Pytorch pretrained weights and return the loaded model."""
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, PYTORCH_WEIGHTS_NAME)
        else:
            model_file = hf_hub_download(
                repo_id=model_id,
                filename=PYTORCH_WEIGHTS_NAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )

        if "config" not in model_kwargs:
            raise ValueError("Config must be supplied to instantiate model")

        model_kwargs.update(model_kwargs.pop("config"))
        model = hydra.utils.instantiate(model_kwargs)

        state_dict = torch.load(model_file, map_location=torch.device(map_location))
        model.load_state_dict(state_dict, strict=strict)  # type: ignore
        model.eval()  # type: ignore

        return model

    @classmethod
    def get_data_config(
        cls,
        model_id: str,
        revision: str,
        cache_dir: Optional[Union[str, Path]] = None,
        force_download: bool = False,
        proxies: Optional[Dict] = None,
        resume_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
    ):
        """Load data config file."""
        if os.path.isdir(model_id):
            print("Loading data config from local directory")
            data_config_file = os.path.join(model_id, DATA_CONFIG_NAME)
        else:
            data_config_file = hf_hub_download(
                repo_id=model_id,
                filename=DATA_CONFIG_NAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )

        return data_config_file

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        config: dict,
        data_config: Optional[Union[str, Path]],
        repo_id: Optional[str] = None,
        push_to_hub: bool = False,
        wandb_ids: Optional[Union[list[str], str]] = None,
        card_template_path=None,
        **kwargs,
    ) -> Optional[str]:
        """
        Save weights in local directory.

        Args:
            save_directory (`str` or `Path`):
                Path to directory in which the model weights and configuration will be saved.
            config (`dict`):
                Model configuration specified as a key/value dictionary.
            data_config (`str` or `Path`):
                The path to the data config.
            repo_id (`str`, *optional*):
                ID of your repository on the Hub. Used only if `push_to_hub=True`. Will default to
                the folder name if not provided.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Huggingface Hub after saving it.
            wandb_ids: Identifier(s) of the model on wandb.
            card_template_path: Path to the huggingface model card template. Defaults to card in
                PVNet library if set to None.
            kwargs:
                Additional key word arguments passed along to the
                [`~ModelHubMixin._from_pretrained`] method.
        """

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # saving model weights/files
        self._save_pretrained(save_directory)

        # saving model and data config
        if isinstance(config, dict):
            (save_directory / CONFIG_NAME).write_text(json.dumps(config, indent=4))

        # Save cleaned datapipes configuration file
        if data_config is not None:
            make_clean_data_config(data_config, save_directory / DATA_CONFIG_NAME)

        # Creating and saving model card.
        card_data = ModelCardData(language="en", license="mit", library_name="pytorch")
        if card_template_path is None:
            card_template_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/model_card_template.md"
            )

        if isinstance(wandb_ids, str):
            wandb_ids = [wandb_ids]

        wandb_links = ""
        for wandb_id in wandb_ids:
            link = f"https://wandb.ai/openclimatefix/pvnet2.1/runs/{wandb_id}"
            wandb_links += f" - [{link}]({link})\n"

        card = ModelCard.from_template(
            card_data,
            template_path=card_template_path,
            wandb_links=wandb_links,
        )

        (save_directory / "README.md").write_text(str(card))

        if push_to_hub:
            api = HfApi()

            api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=save_directory,
            )

        return None


class BaseModel(pl.LightningModule, PVNetModelHubMixin):
    """Abtstract base class for PVNet submodels"""

    def __init__(
        self,
        history_minutes: int,
        forecast_minutes: int,
        optimizer: AbstractOptimizer,
        output_quantiles: Optional[list[float]] = None,
        target_key: str = "gsp",
        interval_minutes: int = 30,
        timestep_intervals_to_plot: Optional[list[int]] = None,
        use_weighted_loss: bool = False,
    ):
        """Abtstract base class for PVNet submodels.

        Args:
            history_minutes (int): Length of the GSP history period in minutes
            forecast_minutes (int): Length of the GSP forecast period in minutes
            optimizer (AbstractOptimizer): Optimizer
            output_quantiles: A list of float (0.0, 1.0) quantiles to predict values for. If set to
                None the output is a single value.
            target_key: The key of the target variable in the batch
            interval_minutes: The interval in minutes between each timestep in the data
            timestep_intervals_to_plot: Intervals, in timesteps, to plot during training
            use_weighted_loss: Whether to use a weighted loss function
        """
        super().__init__()

        self._optimizer = optimizer
        self._target_key_name = target_key
        self._target_key = BatchKey[f"{target_key}"]
        if timestep_intervals_to_plot is not None:
            for interval in timestep_intervals_to_plot:
                assert type(interval) in [list, tuple] and len(interval) == 2, ValueError(
                    f"timestep_intervals_to_plot must be a list of tuples or lists of length 2, "
                    f"but got {timestep_intervals_to_plot=}"
                )
        self.time_step_intervals_to_plot = timestep_intervals_to_plot

        # Model must have lr to allow tuning
        # This setting is only used when lr is tuned with callback
        self.lr = None

        self.history_minutes = history_minutes
        self.forecast_minutes = forecast_minutes
        self.output_quantiles = output_quantiles
        self.interval_minutes = interval_minutes

        # Number of timestemps for 30 minutely data
        self.history_len = history_minutes // interval_minutes
        self.forecast_len = forecast_minutes // interval_minutes

        self.weighted_losses = WeightedLosses(forecast_length=self.forecast_len)

        self._accumulated_metrics = MetricAccumulator()
        self._accumulated_batches = BatchAccumulator(key_to_keep=self._target_key_name)
        self._accumulated_y_hat = PredAccumulator()
        self._horizon_maes = MetricAccumulator()

        # Store whether the model should use quantile regression or simply predict the mean
        self.use_quantile_regression = self.output_quantiles is not None
        self.use_weighted_loss = use_weighted_loss

        # Store the number of ouput features that the model should predict for
        if self.use_quantile_regression:
            self.num_output_features = self.forecast_len * len(self.output_quantiles)
        else:
            self.num_output_features = self.forecast_len

    def _quantiles_to_prediction(self, y_quantiles):
        """
        Convert network prediction into a point prediction.

        Note:
            Implementation copied from:
                https://pytorch-forecasting.readthedocs.io/en/stable/_modules/pytorch_forecasting
                /metrics/quantile.html#QuantileLoss.loss

        Args:
            y_quantiles: Quantile prediction of network

        Returns:
            torch.Tensor: Point prediction
        """
        # y_quantiles Shape: batch_size, seq_length, num_quantiles
        idx = self.output_quantiles.index(0.5)
        y_median = y_quantiles[..., idx]
        return y_median

    def _calculate_qauntile_loss(self, y_quantiles, y):
        """Calculate quantile loss.

        Note:
            Implementation copied from:
                https://pytorch-forecasting.readthedocs.io/en/stable/_modules/pytorch_forecasting
                /metrics/quantile.html#QuantileLoss.loss

        Args:
            y_quantiles: Quantile prediction of network
            y: Target values

        Returns:
            Quantile loss
        """
        # calculate quantile loss
        losses = []
        for i, q in enumerate(self.output_quantiles):
            errors = y - y_quantiles[..., i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
        losses = 2 * torch.cat(losses, dim=2)
        if self.use_weighted_loss:
            weights = self.weighted_losses.weights.unsqueeze(1).unsqueeze(0).to(y.device)
            losses = losses * weights
        return losses.mean()

    def _calculate_common_losses(self, y, y_hat):
        """Calculate losses common to train, test, and val"""

        losses = {}

        if self.use_quantile_regression:
            losses["quantile_loss"] = self._calculate_qauntile_loss(y_hat, y)
            y_hat = self._quantiles_to_prediction(y_hat)

        # calculate mse, mae
        mse_loss = F.mse_loss(y_hat, y)
        mae_loss = F.l1_loss(y_hat, y)

        # calculate mse, mae with exp weighted loss
        mse_exp = self.weighted_losses.get_mse_exp(output=y_hat, target=y)
        mae_exp = self.weighted_losses.get_mae_exp(output=y_hat, target=y)

        # TODO: Compute correlation coef using np.corrcoef(tensor with
        # shape (2, num_timesteps))[0, 1] on each example, and taking
        # the mean across the batch?
        losses.update(
            {
                "MSE": mse_loss,
                "MAE": mae_loss,
                "MSE_EXP": mse_exp,
                "MAE_EXP": mae_exp,
            }
        )

        return losses

    def _step_mae_and_mse(self, y, y_hat, dict_key_root):
        """Calculate the MSE and MAE at each forecast step"""
        losses = {}

        mse_each_step = torch.mean((y_hat - y) ** 2, dim=0)
        mae_each_step = torch.mean(torch.abs(y_hat - y), dim=0)

        losses.update({f"MSE_{dict_key_root}/step_{i:03}": m for i, m in enumerate(mse_each_step)})
        losses.update({f"MAE_{dict_key_root}/step_{i:03}": m for i, m in enumerate(mae_each_step)})

        return losses

    def _calculate_val_losses(self, y, y_hat):
        """Calculate additional validation losses"""

        losses = {}

        if self.use_quantile_regression:
            # Add fraction below each quantile for calibration
            for i, quantile in enumerate(self.output_quantiles):
                below_quant = y <= y_hat[..., i]
                # Mask values small values, which are dominated by night
                mask = y >= 0.01
                losses[f"fraction_below_{quantile}_quantile"] = (below_quant[mask]).float().mean()

            # Take median value for remaining metric calculations
            y_hat = self._quantiles_to_prediction(y_hat)

        # Log the loss at each time horizon
        losses.update(self._step_mae_and_mse(y, y_hat, dict_key_root="horizon"))

        # Log the persistance losses
        y_persist = y[:, -1].unsqueeze(1).expand(-1, self.forecast_len)
        losses["MAE_persistence/val"] = F.l1_loss(y_persist, y)
        losses["MSE_persistence/val"] = F.mse_loss(y_persist, y)

        # Log persistance loss at each time horizon
        losses.update(self._step_mae_and_mse(y, y_persist, dict_key_root="persistence"))
        return losses

    def _calculate_test_losses(self, y, y_hat):
        """Calculate additional test losses"""
        # No additional test losses
        losses = {}
        return losses

    def _training_accumulate_log(self, batch, batch_idx, losses, y_hat):
        """Internal function to accumulate training batches and log results.

        This is used when accummulating grad batches. Should make the variability in logged training
        step metrics indpendent on whether we accumulate N batches of size B or just use a larger
        batch size of N*B with no accumulaion.
        """

        losses = {k: v.detach().cpu() for k, v in losses.items()}
        y_hat = y_hat.detach().cpu()

        self._accumulated_metrics.append(losses)
        self._accumulated_batches.append(batch)
        self._accumulated_y_hat.append(y_hat)

        if not self.trainer.fit_loop._should_accumulate():
            losses = self._accumulated_metrics.flush()
            batch = self._accumulated_batches.flush()
            y_hat = self._accumulated_y_hat.flush()

            self.log_dict(
                losses,
                on_step=True,
                on_epoch=True,
            )

            # Number of accumulated grad batches
            grad_batch_num = (batch_idx + 1) / self.trainer.accumulate_grad_batches

            # We only create the figure every 8 log steps
            # This was reduced as it was creating figures too often
            if grad_batch_num % (8 * self.trainer.log_every_n_steps) == 0:
                fig = plot_batch_forecasts(
                    batch,
                    y_hat,
                    batch_idx,
                    quantiles=self.output_quantiles,
                    key_to_plot=self._target_key_name,
                )
                fig.savefig("latest_logged_train_batch.png")
                plt.close(fig)

    def training_step(self, batch, batch_idx):
        """Run training step"""
        y_hat = self(batch)
        y = batch[self._target_key][:, -self.forecast_len :, 0]

        losses = self._calculate_common_losses(y, y_hat)
        losses = {f"{k}/train": v for k, v in losses.items()}

        self._training_accumulate_log(batch, batch_idx, losses, y_hat)

        if self.use_quantile_regression:
            opt_target = losses["quantile_loss/train"]
        else:
            opt_target = losses["MAE/train"]
        return opt_target

    def _log_forecast_plot(self, batch, y_hat, accum_batch_num, timesteps_to_plot, plot_suffix):
        """Log forecast plot to wandb"""
        fig = plot_batch_forecasts(
            batch,
            y_hat,
            quantiles=self.output_quantiles,
            key_to_plot=self._target_key_name,
        )

        plot_name = f"val_forecast_samples/batch_idx_{accum_batch_num}_{plot_suffix}"

        try:
            self.logger.experiment.log({plot_name: wandb.Image(fig)})
        except Exception as e:
            print(f"Failed to log {plot_name} to wandb")
            print(e)
        plt.close(fig)

    def validation_step(self, batch: dict, batch_idx):
        """Run validation step"""
        y_hat = self(batch)
        # Sensor seems to be in batch, station, time order
        y = batch[self._target_key][:, -self.forecast_len :, 0]

        # Expand persistence to be the same shape as y
        losses = self._calculate_common_losses(y, y_hat)
        losses.update(self._calculate_val_losses(y, y_hat))

        # Store these to make horizon accuracy plot
        self._horizon_maes.append(
            {i: losses[f"MAE_horizon/step_{i:03}"].cpu().numpy() for i in range(self.forecast_len)}
        )

        logged_losses = {f"{k}/val": v for k, v in losses.items()}

        self.log_dict(
            logged_losses,
            on_step=False,
            on_epoch=True,
        )

        accum_batch_num = batch_idx // self.trainer.accumulate_grad_batches

        # Make plots only if using wandb logger
        if isinstance(self.logger, pl.loggers.WandbLogger) and accum_batch_num in [0, 1]:
            # Store these temporarily under self
            if not hasattr(self, "_val_y_hats"):
                self._val_y_hats = PredAccumulator()
                self._val_batches = BatchAccumulator(key_to_keep=self._target_key_name)

            self._val_y_hats.append(y_hat)
            self._val_batches.append(batch)

            # if batch has accumulated
            if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
                y_hat = self._val_y_hats.flush()
                batch = self._val_batches.flush()

                self._log_forecast_plot(
                    batch,
                    y_hat,
                    accum_batch_num,
                    timesteps_to_plot=None,
                    plot_suffix="all",
                )

                if self.time_step_intervals_to_plot is not None:
                    for interval in self.time_step_intervals_to_plot:
                        self._log_forecast_plot(
                            batch,
                            y_hat,
                            accum_batch_num,
                            timesteps_to_plot=interval,
                            plot_suffix=f"timestep_{interval}",
                        )

                del self._val_y_hats
                del self._val_batches

        return logged_losses

    def on_validation_epoch_end(self):
        """Run on epoch end"""

        horizon_maes_dict = self._horizon_maes.flush()

        # Create the horizon accuracy curve
        if isinstance(self.logger, pl.loggers.WandbLogger):
            per_step_losses = [[i, horizon_maes_dict[i]] for i in range(self.forecast_len)]
            try:
                table = wandb.Table(data=per_step_losses, columns=["horizon_step", "MAE"])
                wandb.log(
                    {
                        "horizon_loss_curve": wandb.plot.line(
                            table, "horizon_step", "MAE", title="Horizon loss curve"
                        )
                    },
                )
            except Exception as e:
                print("Failed to log horizon_loss_curve to wandb")
                print(e)

    def test_step(self, batch, batch_idx):
        """Run test step"""
        y_hat = self(batch)
        y = batch[self._target_key][:, -self.forecast_len :, 0]

        losses = self._calculate_common_losses(y, y_hat)
        losses.update(self._calculate_val_losses(y, y_hat))
        losses.update(self._calculate_test_losses(y, y_hat))
        logged_losses = {f"{k}/test": v for k, v in losses.items()}

        self.log_dict(
            logged_losses,
            on_step=False,
            on_epoch=True,
        )

        if self.use_quantile_regression:
            y_hat = self._quantiles_to_prediction(y_hat)

        return construct_ocf_ml_metrics_batch_df(batch, y, y_hat)

    def on_test_epoch_end(self, outputs):
        """Evalauate test results using oc_ml_metrics"""
        results_df = pd.concat(outputs)
        # setting model_name="test" gives us keys like "test/mw/forecast_horizon_30_minutes/mae"
        metrics = evaluation(results_df=results_df, model_name="test", outturn_unit="mw")

        self.log_dict(
            metrics,
        )

    def configure_optimizers(self):
        """Configure the optimizers using learning rate found with LR finder if used"""
        if self.lr is not None:
            # Use learning rate found by learning rate finder callback
            self._optimizer.lr = self.lr
        return self._optimizer(self)
