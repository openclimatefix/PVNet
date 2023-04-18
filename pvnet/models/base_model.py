import logging
import wandb

import numpy as np
import pandas as pd
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from typing import Literal

from nowcasting_utils.models.loss import WeightedLosses
from nowcasting_utils.models.metrics import (
    mae_each_forecast_horizon,
    mse_each_forecast_horizon,
)
from ocf_datapipes.utils.consts import BatchKey
from ocf_datapipes.utils.geospatial import osgb_to_lat_lon

from ocf_ml_metrics.evaluation.evaluation import evaluation
from pvnet.utils import construct_ocf_ml_metrics_batch_df, plot_batch_forecasts 

logger = logging.getLogger(__name__)

activities = [torch.profiler.ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(torch.profiler.ProfilerActivity.CUDA)

    
class PredAccumulator:
    """A class for accumulating y-predictions when using grad accumulation and the batch size is
    small. 
    
    Attributes:
        _y_hats (list[torch.Tensor]): List of prediction tensors
    """
    def __init__(self):
        self._y_hats = []

    def __bool__(self):
        return len(self._y_hats)>0
        
    def append(self, y_hat: torch.Tensor):
        """Append a sub-batch of predictions"""
        self._y_hats += [y_hat]
    
    def flush(self) -> torch.Tensor:
        y_hat = torch.cat(self._y_hats, dim=0)
        self._y_hats = []
        return y_hat

    
class DictListAccumulator:
    
    @staticmethod
    def dict_list_append(d1, d2):
        for k, v in d2.items():
            d1[k] += [v]

    @staticmethod
    def dict_init_list(d):
        return {k:[v] for k, v in d.items()}
    

class MetricAccumulator(DictListAccumulator):
    """A class for accumulating, and finding the mean of logging metrics when using grad 
    accumulation and the batch size is small. 
    
    Attributes:
        _metrics (Dict[str, list[float]]): Dictionary containing lists of metrics.
    """
    
    def __init__(self):
        self._metrics = {}
        
    def __bool__(self):
        return self._metrics!={}
    
    def append(self, loss_dict: dict[str, float]):
        if not self:
            self._metrics = self.dict_init_list(loss_dict)
        else:
            self.dict_list_append(self._metrics, loss_dict)
            
    def flush(self) -> dict[str, float]:
        mean_metrics = {k:np.mean(v) for k, v in self._metrics.items()}
        self._metrics = {}
        return mean_metrics
        

class BatchAccumulator(DictListAccumulator):
    """A class for accumulating batches when using grad accumulation and the batch size is small. 
    
    Attributes:
        _batches (Dict[BatchKey, list[torch.Tensor]]): Dictionary containing lists of metrics.
    """
    
    def __init__(self):
        self._batches = {}
        
    def __bool__(self):
        return self._batches!={}
        
    @staticmethod
    def filter_batch_dict(d):
        keep_keys = [BatchKey.gsp, BatchKey.gsp_id, BatchKey.gsp_t0_idx, BatchKey.gsp_time_utc]
        return {k:v for k, v in d.items() if k in keep_keys}
    
    def append(self, batch: dict[BatchKey, list[torch.Tensor]]):
        if not self:
            self._batches = self.dict_init_list(self.filter_batch_dict(batch))
        else:
            self.dict_list_append(self._batches, self.filter_batch_dict(batch))
            
    def flush(self) -> dict[BatchKey, list[torch.Tensor]]:
        batch = {}
        for k, v in self._batches.items():
            if k==BatchKey.gsp_t0_idx:
                batch[k] = v[0]
            else:
                batch[k] = torch.cat(v, dim=0)
        self._batches = {}
        return batch
                    

        
class BaseModel(pl.LightningModule):

    def __init__(
            self, 
            history_minutes, 
            forecast_minutes, 
            optimizer,
    ):
        super().__init__()
        
        self._optimizer = optimizer
        
        # Model must have lr to allow tuning
        # This setting is only used when lr is tuned with callback
        self.lr = None
                
        self.history_minutes = history_minutes
        self.forecast_minutes = forecast_minutes
        
        # Number of timestemps for 5 minutely data
        self.history_len_5 = history_minutes // 5  
        self.forecast_len_5 = forecast_minutes // 5
        
        # Number of timestemps for 30 minutely data
        self.history_len_30 = history_minutes // 30
        self.forecast_len_30 = forecast_minutes // 30

        # Number of timesteps for 60 minutely data
        # Note that ceil is taken as for 30 minutes of history data, one history value will be used
        self.history_len_60 = int(np.ceil(history_minutes / 60))
        self.forecast_len_60 = self.forecast_minutes // 60

        self.forecast_len = self.forecast_len_30
        self.history_len = self.history_len_30
        
        self.weighted_losses = WeightedLosses(forecast_length=self.forecast_len)
        
        self._accumulated_metrics = MetricAccumulator()
        self._accumulated_batches = BatchAccumulator()
        self._accumulated_y_hat = PredAccumulator()
        
        
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
        mse_each_step = mse_each_forecast_horizon(output=y_hat, target=y)
        mae_each_step = mae_each_forecast_horizon(output=y_hat, target=y)

        losses = {
            f"MSE_horizon/step_{i:02}": m for i, m in enumerate(mse_each_step)
        }
        losses.update(
            {
                f"MAE_horizon/step_{i:02}": m for i, m in enumerate(mae_each_step)
            }
        )
        return losses
    
        
    def _calculate_test_losses(self, y, y_hat):
        """Calculate additional test losses"""
        # No additional test losses
        losses = {}
        return losses
            

    def _training_accumulate_log(self, batch, batch_idx, losses, y_hat):
        """Internal function to accumulate training batches and log results when
        using accummulated grad batches. Should make the variability in logged training step metrics
        indpendent on whether we accumulate N batches of size B or just use a larger batch size of
        N*B with no accumulaion.
        """
        
        losses = {k:v.detach().cpu() for k, v in losses.items()}
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
            grad_batch_num = (batch_idx+1)/self.trainer.accumulate_grad_batches
            
            # We only create the figure every 8 log steps
            # This was reduced as it was creating figures too often
            if grad_batch_num%(8*self.trainer.log_every_n_steps)==0:
                fig = plot_batch_forecasts(batch, y_hat, batch_idx)
                fig.savefig(f"latest_logged_train_batch.png")

        
    def training_step(self, batch, batch_idx):
        
        y_hat = self(batch)
        y = batch[BatchKey.gsp][:, -self.forecast_len:, 0]
        
        losses = self._calculate_common_losses(y, y_hat)
        losses = {f"{k}/train":v for k, v in losses.items()}
        
        self._training_accumulate_log(batch, batch_idx, losses, y_hat)
        
        return losses["MAE/train"]
    
    def validation_step(self, batch: dict, batch_idx):
        # put the batch data through the model
        y_hat = self(batch)
        y = batch[BatchKey.gsp][:, -self.forecast_len:, 0]
        
        losses = self._calculate_common_losses(y, y_hat)
        losses.update(self._calculate_val_losses(y, y_hat))
        
        logged_losses = {f"{k}/val":v for k, v in losses.items()}

        self.log_dict(
            logged_losses,
            on_step=False, 
            on_epoch=True,
        )
        
        
        accum_batch_num = batch_idx//self.trainer.accumulate_grad_batches
        
        if accum_batch_num in [0, 1]:
            
            # Store these temporarily under self
            if not hasattr(self, "_val_y_hats"):
                self._val_y_hats = PredAccumulator()
                self._val_batches = BatchAccumulator()
            
            self._val_y_hats.append(y_hat)
            self._val_batches.append(batch)
            
            # if batch had accumulated
            if (batch_idx+1)%self.trainer.accumulate_grad_batches==0:

                y_hat = self._val_y_hats.flush()
                batch = self._val_batches.flush()
                
                fig = plot_batch_forecasts(batch, y_hat)
                
                self.logger.experiment.log({
                    f"val_forecast_samples/batch_idx_{accum_batch_num}": wandb.Image(fig),
                })
                del self._val_y_hats
                del self._val_batches

        return logged_losses        
        

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
            on_step=False, 
            on_epoch=True,
        )
        
        return construct_ocf_ml_metrics_batch_df(batch, y, y_hat)
    
    def on_test_epoch_end(self, outputs):
        results_df = pd.concat(outputs)
        # setting model_name="test" gives us keys like "test/mw/forecast_horizon_30_minutes/mae"
        metrics = evaluation(results_df=results_df, model_name="test", outturn_unit='mw')
        
        self.log_dict(
            metrics,
        )
        
    def configure_optimizers(self):
        if self.lr is not None:
            # Use learning rate found by learning rate finder callback
            self._optimizer.lr = self.lr
        return self._optimizer(self.parameters())