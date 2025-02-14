"""The default composite model architecture for PVNet"""

import glob
from collections import OrderedDict
from typing import Optional

import hydra
import torch
import torch.nn.functional as F
from pyaml_env import parse_config
from torch import nn

import pvnet
from pvnet.models.multimodal.linear_networks.basic_blocks import AbstractLinearNetwork
from pvnet.models.multimodal.multimodal_base import MultimodalBaseModel
from pvnet.optimizers import AbstractOptimizer


class Model(MultimodalBaseModel):
    """Neural network which combines information from different sources

    The network is trained via unimodal teachers [1].

    Architecture is roughly as follows:

    - Satellite data, if included, is put through an encoder which transforms it from 4D, with time,
        channel, height, and width dimensions to become a 1D feature vector.
    - NWP, if included, is put through a similar encoder.
    - PV site-level data, if included, is put through an encoder which transforms it from 2D, with
        time and system-ID dimensions, to become a 1D feature vector.
    - The satellite features*, NWP features*, PV site-level features*, GSP ID embedding*, and sun
        paramters* are concatenated into a 1D feature vector and passed through another neural
        network to combine them and produce a forecast.

    * if included
    [1] https://arxiv.org/pdf/2305.01233.pdf
    """

    name = "unimodal_teacher"

    def __init__(
        self,
        output_network: AbstractLinearNetwork,
        output_quantiles: Optional[list[float]] = None,
        include_gsp_yield_history: bool = True,
        include_sun: bool = True,
        embedding_dim: Optional[int] = 16,
        forecast_minutes: int = 30,
        history_minutes: int = 60,
        optimizer: AbstractOptimizer = pvnet.optimizers.Adam(),
        mode_teacher_dict: dict = {},
        val_best: bool = True,
        cold_start: bool = True,
        enc_loss_frac: float = 0.3,
        adapt_batches: Optional[bool] = False,
    ):
        """Neural network which combines information from different sources.

        The network is trained via unimodal teachers [1].

        [1] https://arxiv.org/pdf/2305.01233.pdf

        Notes:
            In the args, where it says a module `m` is partially instantiated, it means that a
            normal pytorch module will be returned by running `mod = m(**kwargs)`. In this library,
            this partial instantiation is generally achieved using partial instantiation via hydra.
            However, the arg is still valid as long as `m(**kwargs)` returns a valid pytorch module
            - for example if `m` is a regular function.

        Args:
            output_network: A partially instatiated pytorch Module class used to combine the 1D
                features to produce the forecast.
            output_quantiles: A list of float (0.0, 1.0) quantiles to predict values for. If set to
                None the output is a single value.
            include_gsp_yield_history: Include GSP yield data.
            include_sun: Include sun azimuth and altitude data.
            embedding_dim: Number of embedding dimensions to use for GSP ID. Not included if set to
                `None`.
            forecast_minutes: The amount of minutes that should be forecasted.
            history_minutes: The default amount of historical minutes that are used.
            optimizer: Optimizer factory function used for network.
            mode_teacher_dict: A dictionary of paths to different model checkpoint directories,
                which will be used as the unimodal teachers.
            val_best: Whether to load the model which performed best on the validation set. Else the
                last checkpoint is loaded.
            cold_start: Whether to train the uni-modal encoders from scratch. Else start them with
                weights from the uni-modal teachers.
            enc_loss_frac: Fraction of total loss attributed to the teacher encoders.
            adapt_batches: If set to true, we attempt to slice the batches to the expected shape for
                the model to use. This allows us to overprepare batches and slice from them for the
                data we need for a model run.
        """

        self.include_gsp_yield_history = include_gsp_yield_history
        self.include_sun = include_sun
        self.embedding_dim = embedding_dim
        self.enc_loss_frac = enc_loss_frac
        self.include_sat = False
        self.include_nwp = False
        self.include_pv = False
        self.adapt_batches = adapt_batches

        # This is set but modified later based on the teachers
        self.add_image_embedding_channel = False

        super().__init__(
            history_minutes=history_minutes,
            forecast_minutes=forecast_minutes,
            optimizer=optimizer,
            output_quantiles=output_quantiles,
            target_key="gsp",
        )

        # Number of features expected by the output_network
        # Add to this as network pices are constructed
        fusion_input_features = 0

        self.teacher_models = torch.nn.ModuleDict()
        self.mode_teacher_dict = mode_teacher_dict

        for mode, path in mode_teacher_dict.items():
            # load teacher model and freeze its weights
            self.teacher_models[mode] = self.get_unimodal_encoder(path, True, val_best=val_best)

            for param in self.teacher_models[mode].parameters():
                param.requires_grad = False

            # Recreate model as student
            mode_student_model = self.get_unimodal_encoder(
                path, load_weights=(not cold_start), val_best=val_best
            )

            if mode == "sat":
                self.include_sat = True
                self.sat_sequence_len = mode_student_model.sat_sequence_len
                self.sat_encoder = mode_student_model.sat_encoder

                if mode_student_model.add_image_embedding_channel:
                    self.sat_embed = mode_student_model.sat_embed
                    self.add_image_embedding_channel = True

                fusion_input_features += self.sat_encoder.out_features

            elif mode == "site":
                self.include_pv = True
                self.site_encoder = mode_student_model.site_encoder
                fusion_input_features += self.site_encoder.out_features

            elif mode.startswith("nwp"):
                nwp_source = mode.removeprefix("nwp/")

                if not self.include_nwp:
                    self.include_nwp = True
                    self.nwp_encoders_dict = torch.nn.ModuleDict()

                    if mode_student_model.add_image_embedding_channel:
                        self.add_image_embedding_channel = True
                        self.nwp_embed_dict = torch.nn.ModuleDict()

                self.nwp_encoders_dict[nwp_source] = mode_student_model.nwp_encoders_dict[
                    nwp_source
                ]

                if self.add_image_embedding_channel:
                    self.nwp_embed_dict[nwp_source] = mode_student_model.nwp_embed_dict[nwp_source]

                fusion_input_features += self.nwp_encoders_dict[nwp_source].out_features

        if self.embedding_dim:
            self.embed = nn.Embedding(num_embeddings=318, embedding_dim=embedding_dim)
            fusion_input_features += embedding_dim

        if self.include_sun:
            self.sun_fc1 = nn.Linear(
                in_features=2 * (self.forecast_len + self.history_len + 1),
                out_features=16,
            )
            fusion_input_features += 16

        if include_gsp_yield_history:
            fusion_input_features += self.history_len

        self.output_network = output_network(
            in_features=fusion_input_features,
            out_features=self.num_output_features,
        )

        self.save_hyperparameters()

    def get_unimodal_encoder(self, path, load_weights, val_best):
        """Load a model to function as a unimodal teacher"""

        model_config = parse_config(f"{path}/model_config.yaml")

        # Load the teacher model
        encoder = hydra.utils.instantiate(model_config)

        if load_weights:
            if val_best:
                # Only one epoch (best) saved per model
                files = glob.glob(f"{path}/epoch*.ckpt")
                assert len(files) == 1
                checkpoint = torch.load(files[0], map_location="cpu")
            else:
                checkpoint = torch.load(f"{path}/last.ckpt", map_location="cpu")

            encoder.load_state_dict(state_dict=checkpoint["state_dict"])
        return encoder

    def teacher_forward(self, x):
        """Run the teacher models and return their encodings"""
        modes = OrderedDict()
        for mode, teacher_model in self.teacher_models.items():
            # ******************* Satellite imagery *************************
            if mode == "sat":
                # Shape: batch_size, seq_length, channel, height, width
                sat_data = x["satellite_actual"][:, : teacher_model.sat_sequence_len]
                sat_data = torch.swapaxes(sat_data, 1, 2).float()  # switch time and channels

                if self.add_image_embedding_channel:
                    id = x["gsp_id"].int()
                    sat_data = teacher_model.sat_embed(sat_data, id)

                modes[mode] = teacher_model.sat_encoder(sat_data)

            # *********************** NWP Data ************************************
            if mode.startswith("nwp"):
                nwp_source = mode.removeprefix("nwp/")

                # shape: batch_size, seq_len, n_chans, height, width
                nwp_data = x["nwp"][nwp_source]["nwp"].float()
                nwp_data = torch.swapaxes(nwp_data, 1, 2)  # switch time and channels
                nwp_data = torch.clip(nwp_data, min=-50, max=50)
                if teacher_model.add_image_embedding_channel:
                    id = x["gsp_id"].int()
                    nwp_data = teacher_model.nwp_embed_dict[nwp_source](nwp_data, id)

                nwp_out = teacher_model.nwp_encoders_dict[nwp_source](nwp_data)
                modes[mode] = nwp_out

            # *********************** PV Data *************************************
            # Add site-level PV yield
            if mode == "site":
                modes[mode] = teacher_model.site_encoder(x)

        return modes

    def forward(self, x, return_modes=False):
        """Run model forward"""

        if self.adapt_batches:
            x = self._adapt_batch(x)

        modes = OrderedDict()
        # ******************* Satellite imagery *************************
        if self.include_sat:
            # Shape: batch_size, seq_length, channel, height, width
            sat_data = x["satellite_actual"][:, : self.sat_sequence_len]
            sat_data = torch.swapaxes(sat_data, 1, 2).float()  # switch time and channels

            if self.add_image_embedding_channel:
                id = x["gsp_id"].int()
                sat_data = self.sat_embed(sat_data, id)
            modes["sat"] = self.sat_encoder(sat_data)

        # *********************** NWP Data ************************************
        if self.include_nwp:
            # Loop through potentially many NMPs
            for nwp_source in self.nwp_encoders_dict:
                # shape: batch_size, seq_len, n_chans, height, width
                nwp_data = x["nwp"][nwp_source]["nwp"].float()
                nwp_data = torch.swapaxes(nwp_data, 1, 2)  # switch time and channels
                # Some NWP variables can overflow into NaNs when normalised if they have extreme
                # tails
                nwp_data = torch.clip(nwp_data, min=-50, max=50)

                if self.add_image_embedding_channel:
                    id = x["gsp_id"].int()
                    nwp_data = self.nwp_embed_dict[nwp_source](nwp_data, id)

                nwp_out = self.nwp_encoders_dict[nwp_source](nwp_data)
                modes[f"nwp/{nwp_source}"] = nwp_out

        # *********************** PV Data *************************************
        # Add site-level PV yield
        if self.include_pv:
            if self._target_key != "site":
                modes["site"] = self.site_encoder(x)
            else:
                # Target is PV, so only take the history
                pv_history = x["pv"][:, : self.history_len].float()
                modes["site"] = self.site_encoder(pv_history)

        # *********************** GSP Data ************************************
        # add gsp yield history
        if self.include_gsp_yield_history:
            gsp_history = x["gsp"][:, : self.history_len].float()
            gsp_history = gsp_history.reshape(gsp_history.shape[0], -1)
            modes["gsp"] = gsp_history

        # ********************** Embedding of GSP ID ********************
        if self.embedding_dim:
            id = x["gsp_id"].int()
            id_embedding = self.embed(id)
            modes["id"] = id_embedding

        if self.include_sun:
            sun = torch.cat(
                (
                    x["gsp_solar_azimuth"],
                    x["gsp_solar_elevation"],
                ),
                dim=1,
            ).float()
            sun = self.sun_fc1(sun)
            modes["sun"] = sun

        out = self.output_network(modes)

        if self.use_quantile_regression:
            # Shape: batch_size, seq_length * num_quantiles
            out = out.reshape(out.shape[0], self.forecast_len, len(self.output_quantiles))

        if return_modes:
            return out, modes
        else:
            return out

    def _calculate_teacher_loss(self, modes, teacher_modes):
        enc_losses = {}
        for m, enc in teacher_modes.items():
            enc_losses[f"enc_loss/{m}"] = F.l1_loss(enc, modes[m])
        enc_losses["enc_loss/total"] = sum([v for k, v in enc_losses.items()])
        return enc_losses

    def training_step(self, batch, batch_idx):
        """Run training step"""
        y_hat, modes = self.forward(batch, return_modes=True)
        y = batch[self._target_key][:, -self.forecast_len :, 0]

        losses = self._calculate_common_losses(y, y_hat)

        teacher_modes = self.teacher_forward(batch)
        teacher_loss = self._calculate_teacher_loss(modes, teacher_modes)
        losses.update(teacher_loss)

        if self.use_quantile_regression:
            opt_target = losses["quantile_loss"]
        else:
            opt_target = losses["MAE"]

        t_loss = teacher_loss["enc_loss/total"]

        # The scales of the two losses
        l_s = opt_target.detach()
        tl_s = max(t_loss.detach(), 1e-9)

        # opt_target = t_loss/tl_s * l_s * self.enc_loss_frac + opt_target * (1-self.enc_loss_frac)
        losses["opt_loss"] = t_loss / tl_s * l_s * self.enc_loss_frac + opt_target * (
            1 - self.enc_loss_frac
        )

        losses = {f"{k}/train": v for k, v in losses.items()}
        self._training_accumulate_log(batch, batch_idx, losses, y_hat)

        return losses["opt_loss/train"]

    def convert_to_multimodal_model(self, config):
        """Convert the model into a multimodal model class whilst preserving weights"""
        config = config.copy()

        if "cold_start" in config:
            del config["cold_start"]

        config["_target_"] = "pvnet.models.multimodal.multimodal.Model"

        sources = []
        for mode, path in config["mode_teacher_dict"].items():
            model_config = parse_config(f"{path}/model_config.yaml")

            if mode.startswith("nwp"):
                nwp_source = mode.removeprefix("nwp/")
                if "nwp_encoders_dict" in config:
                    for key in ["nwp_encoders_dict", "nwp_history_minutes", "nwp_forecast_minutes"]:
                        config[key][nwp_source] = model_config[key][nwp_source]
                    sources.append("nwp")
                else:
                    for key in ["nwp_encoders_dict", "nwp_history_minutes", "nwp_forecast_minutes"]:
                        config[key] = {nwp_source: model_config[key][nwp_source]}
                config["add_image_embedding_channel"] = model_config["add_image_embedding_channel"]

            elif mode == "sat":
                for key in [
                    "sat_encoder",
                    "add_image_embedding_channel",
                    "min_sat_delay_minutes",
                    "sat_history_minutes",
                ]:
                    config[key] = model_config[key]
                sources.append("sat")

            elif mode == "site":
                for key in ["site_encoder", "site_history_minutes"]:
                    config[key] = model_config[key]
                sources.append("site")

        del config["mode_teacher_dict"]

        # Load the teacher model
        multimodal_model = hydra.utils.instantiate(config)

        if "sat" in sources:
            multimodal_model.sat_encoder.load_state_dict(self.sat_encoder.state_dict())
        if "nwp" in sources:
            multimodal_model.nwp_encoders_dict.load_state_dict(self.nwp_encoders_dict.state_dict())
        if "site" in sources:
            multimodal_model.site_encoder.load_state_dict(self.site_encoder.state_dict())

        multimodal_model.output_network.load_state_dict(self.output_network.state_dict())

        if self.embedding_dim:
            multimodal_model.embed.load_state_dict(self.embed.state_dict())

        if self.include_sun:
            multimodal_model.sun_fc1.load_state_dict(self.sun_fc1.state_dict())

        return multimodal_model, config
