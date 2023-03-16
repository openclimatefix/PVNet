import logging

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from typing import Literal

# from nowcasting_dataset.data_sources.nwp.nwp_data_source import NWP_VARIABLE_NAMES
from nowcasting_utils.metrics.validation import (
    make_validation_results,
    save_validation_results_to_logger,
)
from nowcasting_utils.models.loss import WeightedLosses
from nowcasting_utils.models.metrics import (
    mae_each_forecast_horizon,
    mse_each_forecast_horizon,
)
from nowcasting_utils.visualization.line import plot_batch_results
from ocf_datapipes.utils.consts import BatchKey

# from nowcasting_utils.visualization.visualization import plot_example

logger = logging.getLogger(__name__)

activities = [torch.profiler.ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(torch.profiler.ProfilerActivity.CUDA)


class BaseModel(pl.LightningModule):

    # results file name
    results_file_name = "results_epoch"

    # list of results dataframes. This is used to save validation results
    results_dfs = []

    def __init__(self):
        super().__init__()

        self.history_len_5 = (
            self.history_minutes // 5
        )  # the number of historic timestemps for 5 minutes data
        self.forecast_len_5 = (
            self.forecast_minutes // 5
        )  # the number of forecast timestemps for 5 minutes data

        self.history_len_30 = (
            self.history_minutes // 30
        )  # the number of historic timestemps for 5 minutes data
        self.forecast_len_30 = (
            self.forecast_minutes // 30
        )  # the number of forecast timestemps for 5 minutes data

        # the number of historic timesteps for 60 minutes data
        # Note that ceil is taken as for 30 minutes of history data, one history value will be used
        self.history_len_60 = int(np.ceil(self.history_minutes / 60))
        self.forecast_len_60 = (
            self.forecast_minutes // 60
        )  # the number of forecast timestemps for 60 minutes data

        self.forecast_len = self.forecast_len_30
        self.history_len = self.history_len_30
        
        self.weighted_losses = WeightedLosses(forecast_length=self.forecast_len)
        
        
    def _calculate_common_losses(self, y, y_hat):
        """Calculate losses common to train, test, and val"""
        
        # calculate mse, mae
        mse_loss = F.mse_loss(y_hat, y)
        mae_loss = F.l1_loss(y_hat, y)

        # calculate mse, mae with exp weighted loss
        mse_exp = self.weighted_losses.get_mse_exp(output=y_hat, target=y)
        mae_exp = self.weighted_losses.get_mae_exp(output=y_hat, target=y)

        # TODO: Compute correlation coef using np.corrcoef(tensor with
        # shape (2, num_timesteps))[0, 1] on each example, and taking
        # the mean across the batch?
        losses = {
                "MSE": mse_loss,
                "MAE": mae_loss,
                "MSE_EXP": mse_exp,
                "MAE_EXP": mae_exp,
        }
        
        return losses
    
    def _calculate_val_losses(self, y, y_hat):
        """Calculate additional validation losses"""
        mse_each_forecast_horizon_metric = mse_each_forecast_horizon(output=y_hat, target=y)
        mae_each_forecast_horizon_metric = mae_each_forecast_horizon(output=y_hat, target=y)

        losses = {
            f"MSE(step={i})": m for i, m in enumerate(mse_each_forecast_horizon_metric)
        }
        losses.update(
            {
                f"MAE(step={i})": m for i, m in enumerate(mae_each_forecast_horizon_metric)
            }
        )
        return losses
        
    def _calculate_test_losses(self, y, y_hat):
        """Calculate additional test losses"""
        # No additional test losses
        losses = {}
        return losses
    
    def _create_val_plot(self, y, y_hat, batch, batch_idx):
        name = f"val/plot/epoch_{self.current_epoch}_{batch_idx}"
        
        y = y.cpu().numpy()
        y_hat = y_hat.cpu().numpy()
        
        if batch_idx in [0, 1, 2, 3, 4]:
            
            # GSP history and future
            gsp_x_and_y = batch[BatchKey.gsp][:, :, 0].cpu().numpy()

            time = [
                pd.to_datetime(x, unit="ns")
                for x in batch[BatchKey.gsp_time_utc].cpu().numpy()
            ]
            time_hat = [t[-self.forecast_len:] for t in time]

            # plot and save to logger
            fig = plot_batch_results(
                model_name=self.name, 
                y=gsp_x_and_y, 
                y_hat=y_hat, 
                x=time, 
                x_hat=time_hat
            )
            fig.write_html(f"temp_{batch_idx}.html")
            try:
                self.logger.experiment[-1][name].upload(f"temp_{batch_idx}.html")
            except Exception:
                pass

        # save validation results
        capacity = (
            batch[BatchKey.gsp_capacity_megawatt_power][:, 0:1]
        ).cpu().numpy()
        
        y_hat_abs = capacity*y_hat
        y_abs = capacity*y

        results = make_validation_results(
            truths_mw=y_abs,
            predictions_mw=y_hat_abs,
            capacity_mwp=capacity.squeeze(),
            gsp_ids=batch[BatchKey.gsp_id][:, 0].cpu().numpy(),
            batch_idx=batch_idx,
            t0_datetimes_utc=pd.to_datetime(batch[BatchKey.gsp_time_utc][:, 0].cpu().numpy()),
        )

        # append so in 'validation_epoch_end' the file is saved
        if batch_idx == 0:
            self.results_dfs = []
        self.results_dfs.append(results)


    def training_step(self, batch, batch_idx):
        
        # put the batch data through the model
        y_hat = self(batch)
        y = batch[BatchKey.gsp][:, -self.forecast_len:, 0]
        
        losses = self._calculate_common_losses(y, y_hat)
        logged_losses = {f"{k}/train":v for k, v in losses.items()}

        self.log_dict(
            logged_losses,
            on_step=True,
            on_epoch=True,
            sync_dist=True  # Required for distributed training
            # (even multi-GPU on signle machine).
        )
        
        return losses["MAE"]

    def validation_step(self, batch: dict, batch_idx):
        # put the batch data through the model
        y_hat = self(batch)
        y = batch[BatchKey.gsp][:, -self.forecast_len:, 0]
        
        losses = self._calculate_common_losses(y, y_hat)
        losses.update(self._calculate_val_losses(y, y_hat))
        
        logged_losses = {f"{k}/val":v for k, v in losses.items()}

        self.log_dict(
            logged_losses,
            on_step=True,
            on_epoch=True,
            sync_dist=True  # Required for distributed training
            # (even multi-GPU on signle machine).
        )
                        
        self._create_val_plot(y, y_hat, batch, batch_idx)

        return losses["MAE"]

    def validation_epoch_end(self, outputs):

        logger.info("Validation epoch end")

        save_validation_results_to_logger(
            results_dfs=self.results_dfs,
            results_file_name=self.results_file_name,
            current_epoch=self.current_epoch,
            logger=self.logger,
        )

    def test_step(self, batch, batch_idx):
        # put the batch data through the model
        y_hat = self(batch)
        y = batch[BatchKey.gsp][:, -self.forecast_len:, 0]
        
        losses = self._calculate_common_losses(y, y_hat)
        losses.update(self._calculate_val_losses(y, y_hat))
        losses.update(self._calculate_test_losses(y, y_hat))
        logged_losses = {f"{k}/test":v for k, v in losses.items()}
        
        self.log_dict(
            logged_losses,
            on_step=True,
            on_epoch=True,
            sync_dist=True  # Required for distributed training
            # (even multi-GPU on signle machine).
        )
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        return optimizer
