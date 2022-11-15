import logging

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from nowcasting_dataloader.batch import BatchML
from nowcasting_dataset.data_sources.nwp.nwp_data_source import NWP_VARIABLE_NAMES
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
from nowcasting_utils.visualization.visualization import plot_example

logger = logging.getLogger(__name__)

activities = [torch.profiler.ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(torch.profiler.ProfilerActivity.CUDA)

default_output_variable = "pv_yield"


class BaseModel(pl.LightningModule):

    # default batch_size
    batch_size = 32

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

        if not hasattr(self, "output_variable"):
            print("setting")
            self.output_variable = default_output_variable

        if self.output_variable == "pv_yield":
            self.forecast_len = self.forecast_len_5
            self.history_len = self.history_len_5
            self.number_of_samples_per_batch = 128
        else:
            self.forecast_len = self.forecast_len_30
            self.history_len = self.history_len_30
            self.number_of_samples_per_batch = 32
        self.number_of_pv_samples_per_batch = 128

        self.weighted_losses = WeightedLosses(forecast_length=self.forecast_len)

    def _training_or_validation_step(
        self, batch, tag: str, return_model_outputs: bool = False
    ):
        """
        batch: The batch data
        tag: either 'Train', 'Validation' , 'Test'
        """

        if type(batch) == dict:
            batch = BatchML(**batch)

        # put the batch data through the model
        y_hat = self(batch)

        # get the true result out. Select the first data point, as this is the pv system in the center of the image
        if self.output_variable == "gsp_yield":
            y = batch.gsp.gsp_yield
        else:
            y = batch.pv.pv_yield
        y = y[0 : self.batch_size, -self.forecast_len :, 0]

        # calculate mse, mae
        mse_loss = F.mse_loss(y_hat, y)
        nmae_loss = (y_hat - y).abs().mean()

        # calculate mse, mae with exp weighted loss
        mse_exp = self.weighted_losses.get_mse_exp(output=y_hat, target=y)
        mae_exp = self.weighted_losses.get_mae_exp(output=y_hat, target=y)

        # TODO: Compute correlation coef using np.corrcoef(tensor with
        # shape (2, num_timesteps))[0, 1] on each example, and taking
        # the mean across the batch?
        self.log_dict(
            {
                f"MSE/{tag}": mse_loss,
                f"NMAE/{tag}": nmae_loss,
                f"MSE_EXP/{tag}": mse_exp,
                f"MAE_EXP/{tag}": mae_exp,
            },
            on_step=True,
            on_epoch=True,
            sync_dist=True  # Required for distributed training
            # (even multi-GPU on signle machine).
        )

        if tag != "Train":
            # add metrics for each forecast horizon
            mse_each_forecast_horizon_metric = mse_each_forecast_horizon(
                output=y_hat, target=y
            )
            mae_each_forecast_horizon_metric = mae_each_forecast_horizon(
                output=y_hat, target=y
            )

            metrics_mse = {
                f"MSE_forecast_horizon_{i}/{tag}": mse_each_forecast_horizon_metric[i]
                for i in range(self.forecast_len_30)
            }
            metrics_mae = {
                f"MSE_forecast_horizon_{i}/{tag}": mae_each_forecast_horizon_metric[i]
                for i in range(self.forecast_len_30)
            }

            self.log_dict(
                {**metrics_mse, **metrics_mae},
                on_step=True,
                on_epoch=True,
                sync_dist=True  # Required for distributed training
                # (even multi-GPU on signle machine).
            )

        if return_model_outputs:
            return nmae_loss, y_hat
        else:
            return nmae_loss

    def training_step(self, batch, batch_idx):

        if (batch_idx == 0) and (self.current_epoch == 0):
            return self._training_or_validation_step(batch, tag="Train")
        else:
            return self._training_or_validation_step(batch, tag="Train")

    def validation_step(self, batch: BatchML, batch_idx):

        if type(batch) == dict:
            batch = BatchML(**batch)

        # get model outputs
        nmae_loss, model_output = self._training_or_validation_step(
            batch, tag="Validation", return_model_outputs=True
        )

        INTERESTING_EXAMPLES = (1, 5, 6, 7, 9, 11, 17, 19)
        name = f"validation/plot/epoch_{self.current_epoch}_{batch_idx}"
        if batch_idx in [0, 1, 2, 3, 4]:

            # make sure the interesting example doesnt go above the batch size
            INTERESTING_EXAMPLES = (
                i for i in INTERESTING_EXAMPLES if i < self.batch_size
            )

            for example_i in INTERESTING_EXAMPLES:
                # 1. Plot example
                if 0:
                    fig = plot_example(
                        batch,
                        model_output,
                        history_minutes=self.history_len_5 * 5,
                        forecast_minutes=self.forecast_len_5 * 5,
                        nwp_channels=NWP_VARIABLE_NAMES,
                        example_i=example_i,
                        epoch=self.current_epoch,
                        output_variable=self.output_variable,
                    )

                    # save fig to log
                    self.logger.experiment[-1].log_image(name, fig)
                    try:
                        fig.close()
                    except Exception:
                        # could not close figure
                        pass

            # 2. plot summary batch of predictions and results
            # make x,y data
            if self.output_variable == "gsp_yield":
                y = batch.gsp.gsp_yield[0 : self.batch_size, :, 0].cpu().numpy()
            else:
                y = batch.pv.pv_yield[0 : self.batch_size, :, 0].cpu().numpy()
            y_hat = model_output[0 : self.batch_size].cpu().numpy()
            time = [
                pd.to_datetime(x, unit="ns")
                for x in batch.gsp.gsp_datetime_index[0 : self.batch_size].cpu().numpy()
            ]
            time_hat = [
                pd.to_datetime(x, unit="ns")
                for x in batch.gsp.gsp_datetime_index[
                    0 : self.batch_size, self.history_len_30 + 1 :
                ]
                .cpu()
                .numpy()
            ]

            # plot and save to logger
            fig = plot_batch_results(
                model_name=self.name, y=y, y_hat=y_hat, x=time, x_hat=time_hat
            )
            fig.write_html(f"temp_{batch_idx}.html")
            try:
                self.logger.experiment[-1][name].upload(f"temp_{batch_idx}.html")
            except Exception:
                pass

        # save validation results
        capacity = batch.gsp.gsp_capacity[:, -self.forecast_len_30 :, 0].cpu().numpy()
        predictions = model_output.cpu().numpy()
        truths = batch.gsp.gsp_yield[:, -self.forecast_len_30 :, 0].cpu().numpy()
        predictions = predictions * capacity
        truths = truths * capacity

        results = make_validation_results(
            truths_mw=truths,
            predictions_mw=predictions,
            capacity_mwp=capacity,
            gsp_ids=batch.gsp.gsp_id[:, 0].cpu(),
            batch_idx=batch_idx,
            t0_datetimes_utc=pd.to_datetime(batch.metadata.t0_datetime_utc),
        )

        # append so in 'validation_epoch_end' the file is saved
        if batch_idx == 0:
            self.results_dfs = []
        self.results_dfs.append(results)

        return nmae_loss

    def validation_epoch_end(self, outputs):

        logger.info("Validation epoch end")

        save_validation_results_to_logger(
            results_dfs=self.results_dfs,
            results_file_name=self.results_file_name,
            current_epoch=self.current_epoch,
            logger=self.logger,
        )

    def test_step(self, batch, batch_idx):
        self._training_or_validation_step(batch, tag="Test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        return optimizer
