"""Base model class for multimodal model and unimodal teacher"""
from torchvision.transforms.functional import center_crop

from pvnet.models.base_model import BaseModel


class MultimodalBaseModel(BaseModel):
    """Base model class for multimodal model and unimodal teacher"""

    def _adapt_batch(self, batch):
        """Slice batches into appropriate shapes for model

        We make some specific assumptions about the original batch and the derived sliced batch:
        - We are only limiting the future projections. I.e. we are never shrinking the batch from
          the left hand side of the time axis, only slicing it from the right
        - We are only shrinking the spatial crop of the satellite and NWP data

        """

        if "gsp" in batch.keys():
            # Slice off the end of the GSP data
            gsp_len = self.forecast_len + self.history_len + 1
            batch["gsp"] = batch["gsp"][:, :gsp_len]
            batch["gsp_time_utc"] = batch["gsp_time_utc"][:, :gsp_len]

        if self.include_sat:
            # Slice off the end of the satellite data and spatially crop
            # Shape: batch_size, seq_length, channel, height, width
            batch["satellite_actual"] = center_crop(
                batch["satellite_actual"][:, : self.sat_sequence_len],
                output_size=self.sat_encoder.image_size_pixels,
            )

        if self.include_nwp:
            # Slice off the end of the NWP data and spatially crop
            for nwp_source in self.nwp_encoders_dict:
                # shape: batch_size, seq_len, n_chans, height, width
                batch["nwp"][nwp_source]["nwp"] = center_crop(
                    batch["nwp"][nwp_source]["nwp"],
                    output_size=self.nwp_encoders_dict[nwp_source].image_size_pixels,
                )[:, : self.nwp_encoders_dict[nwp_source].sequence_length]

        if self.include_sun:
            # Slice off the end of the solar coords data
            for s in ["solar_azimuth", "solar_elevation"]:
                key = f"{self._target_key}_{s}"
                if key in batch.keys():
                    sun_len = self.forecast_len + self.history_len + 1
                    batch[key] = batch[key][:, :sun_len]

        return batch
