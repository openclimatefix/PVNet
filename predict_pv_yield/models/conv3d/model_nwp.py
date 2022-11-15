import logging

import torch
import torch.nn.functional as F
from torch import nn

from predict_pv_yield.models.base_model import BaseModel
from nowcasting_dataloader.batch import BatchML

logging.basicConfig()
_LOG = logging.getLogger("predict_pv_yield")


class Model(BaseModel):

    name = "conv3d_sat_nwp"

    def __init__(
        self,
        include_pv_or_gsp_yield_history: bool = True,
        include_nwp: bool = True,
        forecast_minutes: int = 30,
        history_minutes: int = 60,
        number_of_conv3d_layers: int = 4,
        conv3d_channels: int = 32,
        nwp_image_size_pixels_height: int = 64,
        number_nwp_channels: int = 10,
        fc1_output_features: int = 128,
        fc2_output_features: int = 128,
        fc3_output_features: int = 64,
        output_variable: str = "gsp_yield",
        embedding_dem: int = 16,
        include_pv_yield_history: int = True,
        include_future_satellite: int = True,
    ):
        """
        3d conv model, that takes in different data streams

        architecture is roughly
        1. nwp time series goes into many 3d convolution layers.
        2. Final convolutional layer goes to full connected layer. This is joined by other data inputs like
        - pv yield
        - time variables
        Then there ~4 fully connected layers which end up forecasting the pv yield / gsp into the future

        include_pv_or_gsp_yield_history: include pv yield data
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
        output_variable: the output variable to be predicted
        number_nwp_channels: The number of nwp channels there are
        include_future_satellite: option to include future satellite images, or not
        """

        self.include_pv_or_gsp_yield_history = include_pv_or_gsp_yield_history
        self.include_nwp = include_nwp
        self.number_of_conv3d_layers = number_of_conv3d_layers
        self.number_of_nwp_features = 128
        self.fc1_output_features = fc1_output_features
        self.fc2_output_features = fc2_output_features
        self.fc3_output_features = fc3_output_features
        self.forecast_minutes = forecast_minutes
        self.history_minutes = history_minutes
        self.output_variable = output_variable
        self.number_nwp_channels = number_nwp_channels
        self.embedding_dem = embedding_dem
        self.include_pv_yield_history = include_pv_yield_history
        self.include_future_satellite = include_future_satellite

        super().__init__()

        conv3d_channels = conv3d_channels

        self.nwp_cnn_output_size = (
            conv3d_channels
            * ((nwp_image_size_pixels_height - 2 * self.number_of_conv3d_layers) ** 2)
            * (self.forecast_len_60 + self.history_len_60 + 1)
        )

        # nwp
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
            in_features=self.nwp_cnn_output_size, out_features=self.fc1_output_features
        )
        self.nwp_fc2 = nn.Linear(
            in_features=self.fc1_output_features, out_features=self.number_of_nwp_features
        )

        if self.embedding_dem:
            self.pv_system_id_embedding = nn.Embedding(
                num_embeddings=940, embedding_dim=self.embedding_dem
            )

        if self.include_pv_yield_history:
            self.pv_fc1 = nn.Linear(
                in_features=self.number_of_pv_samples_per_batch * (self.history_len_5 + 1),
                out_features=128,
            )

        fc3_in_features = self.number_of_nwp_features

        self.fc3 = nn.Linear(in_features=fc3_in_features, out_features=self.fc3_output_features)
        self.fc4 = nn.Linear(in_features=self.fc3_output_features, out_features=self.forecast_len)


    def forward(self, x):

        if type(x) == dict:
            x = BatchML(**x)

        # shape: batch_size, n_chans, seq_len, height, width
        nwp_data = x.nwp.data.float()
        out_nwp = F.relu(self.nwp_conv0(nwp_data))
        for i in range(0, self.number_of_conv3d_layers - 1):
            layer = getattr(self, f"nwp_conv{i + 1}")
            out_nwp = F.relu(layer(out_nwp))

        # fully connected layers
        out_nwp = out_nwp.reshape(nwp_data.shape[0], self.nwp_cnn_output_size)
        out_nwp = F.relu(self.nwp_fc1(out_nwp))
        out = F.relu(self.nwp_fc2(out_nwp))

        # which has shape (batch_size, 128)

        # Fully connected layers.
        out = F.relu(self.fc3(out))
        out = self.fc4(out)

        out = out.reshape(nwp_data.shape[0], self.forecast_len)

        return out
