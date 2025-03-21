"""Base model for all PVNet submodels"""

import json
import logging
import os
import tempfile
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
from ocf_data_sampler.sample.base import copy_batch_to_device

from pvnet.models.utils import (
    BatchAccumulator,
    MetricAccumulator,
    PredAccumulator,
)
from pvnet.optimizers import AbstractOptimizer
from pvnet.utils import plot_batch_forecasts

DATA_CONFIG_NAME = "data_config.yaml"


logger = logging.getLogger(__name__)

activities = [torch.profiler.ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(torch.profiler.ProfilerActivity.CUDA)


def make_clean_data_config(input_path, output_path, placeholder="PLACEHOLDER"):
    """Resave the data config and replace the filepaths with a placeholder.

    Args:
        input_path: Path to input configuration file
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
            if config["input_data"][source]["zarr_path"] != "":
                config["input_data"][source]["zarr_path"] = f"{placeholder}.zarr"

    if "nwp" in config["input_data"]:
        for source in config["input_data"]["nwp"]:
            if config["input_data"]["nwp"][source]["zarr_path"] != "":
                config["input_data"]["nwp"][source]["zarr_path"] = f"{placeholder}.zarr"

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


def minimize_data_config(input_path, output_path, model):
    """Strip out parts of the data config which aren't used by the model

    Args:
        input_path: Path to input configuration file
        output_path: Location to save the output configuration file
        model: The PVNet model object
    """
    with open(input_path) as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)

    if "nwp" in config["input_data"]:
        if not model.include_nwp:
            del config["input_data"]["nwp"]
        else:
            for nwp_source in list(config["input_data"]["nwp"].keys()):
                nwp_config = config["input_data"]["nwp"][nwp_source]

                if nwp_source not in model.nwp_encoders_dict:
                    # If not used, delete this source from the config
                    del config["input_data"]["nwp"][nwp_source]
                else:
                    # Replace the image size
                    nwp_pixel_size = model.nwp_encoders_dict[nwp_source].image_size_pixels
                    nwp_config["image_size_pixels_height"] = nwp_pixel_size
                    nwp_config["image_size_pixels_width"] = nwp_pixel_size

                    # Replace the forecast minutes
                    nwp_config["forecast_minutes"] = (
                        model.nwp_encoders_dict[nwp_source].sequence_length
                        - nwp_config["interval_start_minutes"]
                        / nwp_config["time_resolution_minutes"]
                        - 1
                    ) * nwp_config["time_resolution_minutes"]

    if "satellite" in config["input_data"]:
        if not model.include_sat:
            del config["input_data"]["satellite"]
        else:
            sat_config = config["input_data"]["satellite"]

            # Replace the image size
            sat_pixel_size = model.sat_encoder.image_size_pixels
            sat_config["satellite_image_size_pixels_height"] = sat_pixel_size
            sat_config["satellite_image_size_pixels_width"] = sat_pixel_size

            # Replace the satellite delay
            sat_config["live_delay_minutes"] = model.min_sat_delay_minutes

    if "pv" in config["input_data"]:
        if not model.include_pv:
            del config["input_data"]["pv"]

    if "gsp" in config["input_data"]:
        gsp_config = config["input_data"]["gsp"]

        # Replace the forecast minutes
        gsp_config["forecast_minutes"] = model.forecast_minutes

    with open(output_path, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


class PVNetModelHubMixin(PyTorchModelHubMixin):
    """
    Implementation of [`PyTorchModelHubMixin`] to provide model Hub upload/download capabilities.
    """

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str,
        cache_dir: Optional[Union[str, Path]] = None,
        force_download: bool = False,
        proxies: Optional[Dict] = None,
        resume_download: Optional[bool] = None,
        local_files_only: bool = False,
        token: Union[str, bool, None] = None,
        map_location: str = "cpu",
        strict: bool = False,
    ):
        """Load Pytorch pretrained weights and return the loaded model."""

        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, PYTORCH_WEIGHTS_NAME)
            config_file = os.path.join(model_id, CONFIG_NAME)
        else:
            # load model file
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

            # load config file
            config_file = hf_hub_download(
                repo_id=model_id,
                filename=CONFIG_NAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )

        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        model = hydra.utils.instantiate(config)

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

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights from a Pytorch model to a local directory."""
        model_to_save = self.module if hasattr(self, "module") else self  # type: ignore
        torch.save(model_to_save.state_dict(), save_directory / PYTORCH_WEIGHTS_NAME)

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        config: dict,
        data_config: Optional[Union[str, Path]],
        repo_id: Optional[str] = None,
        push_to_hub: bool = False,
        wandb_repo: Optional[str] = None,
        wandb_ids: Optional[Union[list[str], str]] = None,
        card_template_path=None,
        revision: str = "main",
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
                Whether or not to push your model to the HuggingFace Hub after saving it.
            wandb_repo: Identifier of the repo on wandb.
            wandb_ids: Identifier(s) of the model on wandb.
            card_template_path: Path to the HuggingFace model card template. Defaults to card in
                PVNet library if set to None.
            revision (`str`, *optional*, defaults to `"main"`):
                The revision to push the model to. Only used if `push_to_hub=True`.
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

        # Save cleaned configuration file
        if data_config is not None:
            new_data_config_path = save_directory / DATA_CONFIG_NAME

            # Replace the input filenames with place holders
            make_clean_data_config(data_config, new_data_config_path)

            # Taylor the data config to the model being saved
            minimize_data_config(new_data_config_path, new_data_config_path, self)

        # Get appropriate model card
        model_name = repo_id.split("/")[1]
        if model_name == "windnet_india":
            model_card = "wind_india_model_card_template.md"
        elif model_name == "pvnet_india":
            model_card = "pv_india_model_card_template.md"
        else:
            model_card = "pv_uk_regional_model_card_template.md"

        # Creating and saving model card.
        card_data = ModelCardData(language="en", license="mit", library_name="pytorch")
        if card_template_path is None:
            card_template_path = (
                f"{os.path.dirname(os.path.abspath(__file__))}/model_cards/{model_card}"
            )

        if isinstance(wandb_ids, str):
            wandb_ids = [wandb_ids]

        wandb_links = ""
        for wandb_id in wandb_ids:
            link = f"https://wandb.ai/{wandb_repo}/runs/{wandb_id}"
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
                revision=revision,
            )

        return None


class BaseModel(pl.LightningModule, PVNetModelHubMixin):
    """Abstract base class for PVNet submodels"""

    def __init__(
        self,
        history_minutes: int,
        forecast_minutes: int,
        optimizer: AbstractOptimizer,
        output_quantiles: Optional[list[float]] = None,
        target_key: str = "gsp",
        interval_minutes: int = 30,
        timestep_intervals_to_plot: Optional[list[int]] = None,
        forecast_minutes_ignore: Optional[int] = 0,
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
            forecast_minutes_ignore: Number of forecast minutes to ignore when calculating losses.
                For example if set to 60, the model doesnt predict the first 60 minutes
        """
        super().__init__()

        self._optimizer = optimizer
        self._target_key = target_key
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
        self.forecast_minutes_ignore = forecast_minutes_ignore

        # Number of timestemps for 30 minutely data
        self.history_len = history_minutes // interval_minutes
        self.forecast_len = (forecast_minutes - forecast_minutes_ignore) // interval_minutes
        self.forecast_len_ignore = forecast_minutes_ignore // interval_minutes

        self._accumulated_metrics = MetricAccumulator()
        self._accumulated_batches = BatchAccumulator(key_to_keep=self._target_key)
        self._accumulated_y_hat = PredAccumulator()
        self._horizon_maes = MetricAccumulator()

        # Store whether the model should use quantile regression or simply predict the mean
        self.use_quantile_regression = self.output_quantiles is not None

        # Store the number of ouput features that the model should predict for
        if self.use_quantile_regression:
            self.num_output_features = self.forecast_len * len(self.output_quantiles)
        else:
            self.num_output_features = self.forecast_len

        # save all validation results to array, so we can save these to weights n biases
        self.validation_epoch_results = []

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Method to move custom batches to a given device"""
        return copy_batch_to_device(batch, device)

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

    def _calculate_quantile_loss(self, y_quantiles, y):
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

        return losses.mean()

    def _calculate_common_losses(self, y, y_hat):
        """Calculate losses common to train, and val"""

        losses = {}

        if self.use_quantile_regression:
            losses["quantile_loss"] = self._calculate_quantile_loss(y_hat, y)
            y_hat = self._quantiles_to_prediction(y_hat)

        # calculate mse, mae
        mse_loss = F.mse_loss(y_hat, y)
        mae_loss = F.l1_loss(y_hat, y)

        # TODO: Compute correlation coef using np.corrcoef(tensor with
        # shape (2, num_timesteps))[0, 1] on each example, and taking
        # the mean across the batch?
        losses.update(
            {
                "MSE": mse_loss,
                "MAE": mae_loss,
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
                    key_to_plot=self._target_key,
                )
                fig.savefig("latest_logged_train_batch.png")
                plt.close(fig)

    def training_step(self, batch, batch_idx):
        """Run training step"""
        y_hat = self(batch)
        y = batch[self._target_key][:, -self.forecast_len :]

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
            key_to_plot=self._target_key,
        )

        plot_name = f"val_forecast_samples/batch_idx_{accum_batch_num}_{plot_suffix}"

        try:
            self.logger.experiment.log({plot_name: wandb.Image(fig)})
        except Exception as e:
            print(f"Failed to log {plot_name} to wandb")
            print(e)
        plt.close(fig)

    def _log_validation_results(self, batch, y_hat, accum_batch_num):
        """Append validation results to self.validation_epoch_results"""

        # get truth values, shape (b, forecast_len)
        y = batch[self._target_key][:, -self.forecast_len :]
        y = y.detach().cpu().numpy()
        batch_size = y.shape[0]

        # get prediction values, shape (b, forecast_len, quantiles?)
        y_hat = y_hat.detach().cpu().numpy()

        # get time_utc, shape (b, forecast_len)
        time_utc_key = f"{self._target_key}_time_utc"
        time_utc = batch[time_utc_key][:, -self.forecast_len :].detach().cpu().numpy()

        # get target id and change from (b,1) to (b,)
        id_key = f"{self._target_key}_id"
        target_id = batch[id_key].detach().cpu().numpy()
        target_id = target_id.squeeze()

        for i in range(batch_size):
            y_i = y[i]
            y_hat_i = y_hat[i]
            time_utc_i = time_utc[i]
            target_id_i = target_id[i]

            results_dict = {
                "y": y_i,
                "time_utc": time_utc_i,
            }
            if self.use_quantile_regression:
                results_dict.update(
                    {f"y_quantile_{q}": y_hat_i[:, i] for i, q in enumerate(self.output_quantiles)}
                )
            else:
                results_dict["y_hat"] = y_hat_i

            results_df = pd.DataFrame(results_dict)
            results_df["id"] = target_id_i
            results_df["batch_idx"] = accum_batch_num
            results_df["example_idx"] = i

            self.validation_epoch_results.append(results_df)

    def validation_step(self, batch: dict, batch_idx):
        """Run validation step"""

        accum_batch_num = batch_idx // self.trainer.accumulate_grad_batches

        y_hat = self(batch)

        y = batch[self._target_key][:, -self.forecast_len :]

        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            self._log_validation_results(batch, y_hat, accum_batch_num)

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

        # Make plots only if using wandb logger
        if isinstance(self.logger, pl.loggers.WandbLogger) and accum_batch_num in [0, 1]:
            # Store these temporarily under self
            if not hasattr(self, "_val_y_hats"):
                self._val_y_hats = PredAccumulator()
                self._val_batches = BatchAccumulator(key_to_keep=self._target_key)

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

        try:
            # join together validation results, and save to wandb
            validation_results_df = pd.concat(self.validation_epoch_results)
            validation_results_df["error"] = (
                validation_results_df["y"] - validation_results_df["y_quantile_0.5"]
            )

            if isinstance(self.logger, pl.loggers.WandbLogger):
                # log error distribution metrics
                wandb.log(
                    {
                        "2nd_percentile_median_forecast_error": validation_results_df[
                            "error"
                        ].quantile(0.02),
                        "5th_percentile_median_forecast_error": validation_results_df[
                            "error"
                        ].quantile(0.05),
                        "95th_percentile_median_forecast_error": validation_results_df[
                            "error"
                        ].quantile(0.95),
                        "98th_percentile_median_forecast_error": validation_results_df[
                            "error"
                        ].quantile(0.98),
                        "95th_percentile_median_forecast_absolute_error": abs(
                            validation_results_df["error"]
                        ).quantile(0.95),
                        "98th_percentile_median_forecast_absolute_error": abs(
                            validation_results_df["error"]
                        ).quantile(0.98),
                    }
                )

            with tempfile.TemporaryDirectory() as tempdir:
                filename = os.path.join(tempdir, f"validation_results_{self.current_epoch}.csv")
                validation_results_df.to_csv(filename, index=False)

                # make and log wand artifact
                validation_artifact = wandb.Artifact(
                    f"validation_results_epoch_{self.current_epoch}", type="dataset"
                )
                validation_artifact.add_file(filename)
                wandb.log_artifact(validation_artifact)
        except Exception as e:
            print("Failed to log validation results to wandb")
            print(e)

        self.validation_epoch_results = []
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

    def configure_optimizers(self):
        """Configure the optimizers using learning rate found with LR finder if used"""
        if self.lr is not None:
            # Use learning rate found by learning rate finder callback
            self._optimizer.lr = self.lr
        return self._optimizer(self)
