import logging

import torch
import torch.nn.functional as F
from ocf_datapipes.utils.consts import BatchKey
from torch import nn

from pvnet.models.base_model import BaseModel

logging.basicConfig()
_LOG = logging.getLogger("pvnet")


class Model(BaseModel):

    name = "conv3d_sat_nwp"

    def __init__(
        self,
        include_gsp_yield_history: bool = True,
        include_nwp: bool = True,
        forecast_minutes: int = 30,
        history_minutes: int = 60,
        number_of_conv3d_layers: int = 4,
        conv3d_channels: int = 32,
        image_size_pixels: int = 64,
        nwp_image_size_pixels_height: int = 64,
        number_sat_channels: int = 12,
        number_nwp_channels: int = 10,
        fc1_output_features: int = 128,
        fc2_output_features: int = 128,
        fc3_output_features: int = 64,
        embedding_dem: int = 16,
        include_future_satellite: int = False,
        live_satellite_images: bool = True,
        include_sun: bool = True,
    ):
        """
        3d conv model, that takes in different data streams

        architecture is roughly
        1. satellite image time series goes into many 3d convolution layers.
        2. nwp time series goes into many 3d convolution layers.
        3. Final convolutional layer goes to full connected layer. This is joined by other data inputs like
        - pv yield
        - time variables
        Then there ~4 fully connected layers which end up forecasting the gsp into the future

        include_gsp_yield_history: include gsp yield data
        include_nwp: include nwp data
        forecast_len: the amount of minutes that should be forecasted
        history_len: the amount of historical minutes that are used
        number_of_conv3d_layers, number of convolution 3d layers that are use
        conv3d_channels, the amount of convolution 3d channels
        image_size_pixels: the input satellite image size
        nwp_image_size_pixels_height: the input nwp image size
        number_sat_channels: number of nwp channels
        fc1_output_features: number of fully connected outputs nodes out of the the first fully connected layer
        fc2_output_features: number of fully connected outputs nodes out of the the second fully connected layer
        fc3_output_features: number of fully connected outputs nodes out of the the third fully connected layer
        number_nwp_channels: The number of nwp channels there are
        include_future_satellite: option to include future satellite images, or not
        """

        self.include_gsp_yield_history = include_gsp_yield_history
        self.include_nwp = include_nwp
        self.number_of_conv3d_layers = number_of_conv3d_layers
        self.number_of_nwp_features = 128
        self.fc1_output_features = fc1_output_features
        self.fc2_output_features = fc2_output_features
        self.fc3_output_features = fc3_output_features
        self.forecast_minutes = forecast_minutes
        self.history_minutes = history_minutes
        self.number_nwp_channels = number_nwp_channels
        self.embedding_dem = embedding_dem
        self.include_future_satellite = include_future_satellite
        self.live_satellite_images = live_satellite_images
        self.number_sat_channels = number_sat_channels
        self.image_size_pixels = image_size_pixels
        self.include_sun = include_sun

        super().__init__()

        if include_future_satellite:
            self.cnn_output_size_time = self.forecast_len_5 + self.history_len_5 + 1
        else:
            self.cnn_output_size_time = self.history_len_5 + 1

        if live_satellite_images:
            # remove the last 12 satellite images (60 minutes) as no available live
            self.cnn_output_size_time = self.cnn_output_size_time - 6
            if self.cnn_output_size_time <= 0:
                assert Exception("Need to use at least 30 mintues of satellite data in the past")

        self.cnn_output_size = (
            conv3d_channels
            * ((image_size_pixels - 2 * self.number_of_conv3d_layers) ** 2)
            * self.cnn_output_size_time
        )

        self.nwp_cnn_output_size = (
            conv3d_channels
            * ((nwp_image_size_pixels_height - 2 * self.number_of_conv3d_layers) ** 2)
            * (self.forecast_len_60 + self.history_len_60 + 1)
        )

        # conv0
        self.sat_conv0 = nn.Conv3d(
            in_channels=self.number_sat_channels,
            out_channels=conv3d_channels,
            kernel_size=(3, 3, 3),
            padding=(1, 0, 0),
        )
        for i in range(0, self.number_of_conv3d_layers - 1):
            layer = nn.Conv3d(
                in_channels=conv3d_channels,
                out_channels=conv3d_channels,
                kernel_size=(3, 3, 3),
                padding=(1, 0, 0),
            )
            setattr(self, f"sat_conv{i + 1}", layer)

        self.fc1 = nn.Linear(
            in_features=self.cnn_output_size, out_features=self.fc1_output_features
        )
        self.fc2 = nn.Linear(
            in_features=self.fc1_output_features, out_features=self.fc2_output_features
        )

        # nwp
        if include_nwp:
            self.nwp_conv0 = nn.Conv3d(
                in_channels=number_nwp_channels,
                out_channels=conv3d_channels,
                kernel_size=(3, 3, 3),
                padding=(1, 0, 0),
            )
            for i in range(0, self.number_of_conv3d_layers - 1):
                layer = nn.Conv3d(
                    in_channels=conv3d_channels,
                    out_channels=conv3d_channels,
                    kernel_size=(3, 3, 3),
                    padding=(1, 0, 0),
                )
                setattr(self, f"nwp_conv{i + 1}", layer)

            self.nwp_fc1 = nn.Linear(
                in_features=self.nwp_cnn_output_size,
                out_features=self.fc1_output_features,
            )
            self.nwp_fc2 = nn.Linear(
                in_features=self.fc1_output_features,
                out_features=self.number_of_nwp_features,
            )

        if self.embedding_dem:
            self.pv_system_id_embedding = nn.Embedding(
                num_embeddings=1000, embedding_dim=self.embedding_dem
            )

        if self.include_sun:
            # the minus 12 is bit of hard coded smudge for pvnet
            self.sun_fc1 = nn.Linear(
                in_features=2 * (self.forecast_len_30 + self.history_len_30 + 1 ),
                out_features=16,
            )

        fc3_in_features = self.fc2_output_features
        if include_gsp_yield_history:
            fc3_in_features += self.history_len_30 + 1
        if include_nwp:
            fc3_in_features += 128
        if self.embedding_dem:
            fc3_in_features += self.embedding_dem
        if self.include_sun:
            fc3_in_features += 16

        self.fc3 = nn.Linear(in_features=fc3_in_features, out_features=self.fc3_output_features)
        self.fc4 = nn.Linear(in_features=self.fc3_output_features, out_features=self.forecast_len)
        # self.fc5 = nn.Linear(in_features=32, out_features=8)
        # self.fc6 = nn.Linear(in_features=8, out_features=1)

    def forward(self, x):
        
        _LOG.info("model forward pass")

        # ******************* Satellite imagery *************************
        # Shape: batch_size, seq_length, channel, height, width
        sat_data = x[BatchKey.satellite_actual]
        batch_size, seq_len, n_chans, height, width = sat_data.shape
        # switch time and channels
        sat_data = torch.swapaxes(sat_data, 1, 2).float()

        if not self.include_future_satellite:
            sat_data = sat_data[:, :, : self.history_len_5 + 1]

        if self.live_satellite_images:
            sat_data = sat_data[:, :, :-6]

        # :) Pass data through the network :)
        out = F.relu(self.sat_conv0(sat_data))
        for i in range(0, self.number_of_conv3d_layers - 1):
            layer = getattr(self, f"sat_conv{i + 1}")
            out = F.relu(layer(out))

        out = out.reshape(batch_size, self.cnn_output_size)

        # Fully connected layers
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))

        # add gsp yield history
        if self.include_gsp_yield_history:
            gsp_yield_history = (
                x[BatchKey.gsp][:, : self.history_len_30 + 1].nan_to_num(nan=0.0).float()
            )

            gsp_yield_history = gsp_yield_history.reshape(
                gsp_yield_history.shape[0],
                gsp_yield_history.shape[1] * gsp_yield_history.shape[2],
            )
            # join up
            out = torch.cat((out, gsp_yield_history), dim=1)

        # *********************** NWP Data ************************************
        if self.include_nwp:

            # shape: batch_size, seq_len, n_chans, height, width
            nwp_data = x[BatchKey.nwp].float()
            nwp_data = torch.swapaxes(nwp_data, 1, 2)

            out_nwp = F.relu(self.nwp_conv0(nwp_data))
            for i in range(0, self.number_of_conv3d_layers - 1):
                layer = getattr(self, f"nwp_conv{i + 1}")
                out_nwp = F.relu(layer(out_nwp))
            
            # fully connected layers
            out_nwp = out_nwp.reshape(batch_size, self.nwp_cnn_output_size)
            out_nwp = F.relu(self.nwp_fc1(out_nwp))
            out_nwp = F.relu(self.nwp_fc2(out_nwp))

            # join with other FC layer
            out = torch.cat((out, out_nwp), dim=1)

        # ********************** Embedding of GSP ID ********************
        if self.embedding_dem:
            id = x[BatchKey.gsp_id][:, 0]
            id = id.type(torch.IntTensor)
            id = id.to(out.device)
            id_embedding = self.pv_system_id_embedding(id)
            out = torch.cat((out, id_embedding), dim=1)

        if self.include_sun:
            sun = torch.cat(
                (x[BatchKey.gsp_solar_azimuth], x[BatchKey.gsp_solar_elevation]), 
                dim=1
            ).float()
            out_sun = self.sun_fc1(sun)
            out = torch.cat((out, out_sun), dim=1)
        
        # Fully connected layers.
        out = F.relu(self.fc3(out))
        out = self.fc4(out)

        out = out.reshape(batch_size, self.forecast_len)

        return out
