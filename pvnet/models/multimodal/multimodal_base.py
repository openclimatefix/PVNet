"""Base model class for multimodal model and unimodal teacher"""
from ocf_datapipes.batch import BatchKey, NWPBatchKey
from torchvision.transforms.functional import center_crop

from pvnet.models.base_model import BaseModel


class MultimodalBaseModel(BaseModel):
    """Base model class for multimodal model and unimodal teacher"""

    def _adapt_batch(self, batch):
        """Slice batches into appropriate shapes for model
        Returns a new batch dictionary with adapted data, leaving the original batch unchanged.

        We make some specific assumptions about the original batch and the derived sliced batch:
        - We are only limiting the future projections. I.e. we are never shrinking the batch from
          the left hand side of the time axis, only slicing it from the right
        - We are only shrinking the spatial crop of the satellite and NWP data

        """
        # Create a copy of the batch to avoid modifying the original
        new_batch = {key: value.copy() for key, value in batch.items()}

        if BatchKey.gsp in new_batch.keys():
            # Slice off the end of the GSP data
            gsp_len = self.forecast_len + self.history_len + 1
            new_batch[BatchKey.gsp] = new_batch[BatchKey.gsp][:, :gsp_len]
            new_batch[BatchKey.gsp_time_utc] = new_batch[BatchKey.gsp_time_utc][:, :gsp_len]

        if self.include_sat:
            # Slice off the end of the satellite data and spatially crop
            # Shape: batch_size, seq_length, channel, height, width
            new_batch[BatchKey.satellite_actual] = center_crop(
                new_batch[BatchKey.satellite_actual][:, : self.sat_sequence_len],
                output_size=self.sat_encoder.image_size_pixels,
            )

        if self.include_nwp:
            # Slice off the end of the NWP data and spatially crop
            for nwp_source in self.nwp_encoders_dict:
                # shape: batch_size, seq_len, n_chans, height, width
                new_batch[BatchKey.nwp][nwp_source][NWPBatchKey.nwp] = center_crop(
                    new_batch[BatchKey.nwp][nwp_source][NWPBatchKey.nwp],
                    output_size=self.nwp_encoders_dict[nwp_source].image_size_pixels,
                )[:, : self.nwp_encoders_dict[nwp_source].sequence_length]

        if self.include_sun:
            # Slice off the end of the solar coords data
            for s in ["solar_azimuth", "solar_elevation"]:
                key = BatchKey[f"{self._target_key_name}_{s}"]
                if key in new_batch.keys():
                    sun_len = self.forecast_len + self.history_len + 1
                    new_batch[key] = new_batch[key][:, :sun_len]

        return new_batch
