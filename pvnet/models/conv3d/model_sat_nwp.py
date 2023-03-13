import logging

import torch
import torch.nn.functional as F
from ocf_datapipes.utils.consts import BatchKey
from torch import nn

from pvnet.models.base_model import BaseModel

logging.basicConfig()
_LOG = logging.getLogger("pvnet")


class NWPSatelliteEncoder(nn.Module):

    def __init__(
        self,
        sequence_length,
        number_of_conv3d_layers: int = 4,
        conv3d_channels: int = 32,
        image_size_pixels: int = 64,
        number_sat_channels: int = 12,
        fc1_output_features: int = 128,
        fc2_output_features: int = 128,
    ):
        """
        3d conv model, that takes ins satellite data
        sequence_length: the time sequence length of the satellite data
        number_of_conv3d_layers, number of convolution 3d layers that are use
        conv3d_channels, the amount of convolution 3d channels
        image_size_pixels: the input satellite image size
        number_sat_channels: number of nwp channels
        fc1_output_features: number of output nodes out of the the first fully connected layer
        fc2_output_features: number of output nodes out of the the final fully connected layer
        """

        super().__init__()


        self.cnn_output_size = (
            conv3d_channels
            * ((image_size_pixels - 2 * number_of_conv3d_layers) ** 2)
            * sequence_length
        )
        
        conv_layers = []
        
        conv_layers += [
            nn.Conv3d(
                in_channels=number_sat_channels,
                out_channels=conv3d_channels,
                kernel_size=(3, 3, 3),
                padding=(1, 0, 0),
            ),
            nn.LeakyReLU(),
        ]
        for i in range(0, number_of_conv3d_layers - 1):
            conv_layers += [
                nn.Conv3d(
                    in_channels=conv3d_channels,
                    out_channels=conv3d_channels,
                    kernel_size=(3, 3, 3),
                    padding=(1, 0, 0),
                ),
                nn.LeakyReLU(),
            ]
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        self.fc1 = nn.Linear(
            in_features=self.cnn_output_size, out_features=fc1_output_features
        )
        self.fc2 = nn.Linear(
            in_features=fc1_output_features, out_features=fc2_output_features
        )

    def forward(self, x):
                
        out = self.conv_layers(x)
        out = out.reshape(x.shape[0], -1)

        # Fully connected layers
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return out


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
        embedding_dim: int = 16,
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
        """

        self.number_of_conv3d_layers = number_of_conv3d_layers
        self.include_gsp_yield_history = include_gsp_yield_history
        self.include_nwp = include_nwp
        self.include_sun = include_sun
        self.embedding_dim = embedding_dim
        
        # These properties needed for BaseModel
        self.history_minutes = history_minutes
        self.forecast_minutes = forecast_minutes
        
        super().__init__()
        
        # TODO: remove this hardcoding
        # We limit the history to have a delay of 15 mins in satellite data
        self.sat_sequence_len = self.history_len_5 + 1 - 3
        
        
        self.sat_encoder = NWPSatelliteEncoder(
            sequence_length=self.sat_sequence_len ,
            number_of_conv3d_layers=number_of_conv3d_layers,
            conv3d_channels=conv3d_channels,
            image_size_pixels=image_size_pixels,
            number_sat_channels=number_sat_channels,
            fc1_output_features=fc1_output_features,
            fc2_output_features=fc2_output_features,
        )

        if include_nwp:
            self.nwp_encoder = NWPSatelliteEncoder(
                sequence_length=self.forecast_len_60 + self.history_len_60 + 1,
                number_of_conv3d_layers=number_of_conv3d_layers,
                conv3d_channels=conv3d_channels,
                image_size_pixels=nwp_image_size_pixels_height,
                number_sat_channels=number_nwp_channels,
                fc1_output_features=fc1_output_features,
                fc2_output_features=fc2_output_features,
            )
        

        if self.embedding_dim:
            self.embed = nn.Embedding(
                num_embeddings=330, embedding_dim=self.embedding_dim
            )

        if self.include_sun:
            # the minus 12 is bit of hard coded smudge for pvnet
            self.sun_fc1 = nn.Linear(
                in_features=2 * (self.forecast_len_30 + self.history_len_30 + 1),
                out_features=16,
            )

        fc3_in_features = fc2_output_features
        if include_nwp:
            fc3_in_features += fc2_output_features
        if include_gsp_yield_history:
            fc3_in_features += self.history_len_30
        if embedding_dim:
            fc3_in_features += embedding_dim
        if include_sun:
            fc3_in_features += 16

        self.fc3 = nn.Linear(in_features=fc3_in_features, out_features=fc3_output_features)
        self.fc4 = nn.Linear(in_features=fc3_output_features, out_features=self.forecast_len)


    def forward(self, x):
        breakpoint()
        # ******************* Satellite imagery *************************
        # Shape: batch_size, seq_length, channel, height, width
        sat_data = x[BatchKey.satellite_actual]
        batch_size, seq_len, n_chans, height, width = sat_data.shape
        # switch time and channels
        sat_data = torch.swapaxes(sat_data, 1, 2).float()

        sat_data = sat_data[:, :, : self.sat_sequence_len]

        out = self.sat_encoder(sat_data)
        
        # *********************** GSP Data ************************************
        # add gsp yield history
        if self.include_gsp_yield_history:
            gsp_history = x[BatchKey.gsp][:, : self.history_len_30].float()

            gsp_history = gsp_history.reshape(gsp_history.shape[0], -1)
            
            out = torch.cat((out, gsp_history), dim=1)

        # *********************** NWP Data ************************************
        if self.include_nwp:

            # shape: batch_size, seq_len, n_chans, height, width
            nwp_data = x[BatchKey.nwp].float()
            nwp_data = torch.swapaxes(nwp_data, 1, 2)

            nwp_out = self.nwp_encoder(nwp_data)

            # join with other FC layer
            out = torch.cat((out, nwp_out), dim=1)

        # ********************** Embedding of GSP ID ********************
        if self.embedding_dim:
            id = x[BatchKey.gsp_id][:, 0]
            id = id.type(torch.IntTensor)
            id = id.to(out.device)
            id_embedding = self.embed(id)
            out = torch.cat((out, id_embedding), dim=1)

        if self.include_sun:
            sun = torch.cat(
                (x[BatchKey.gsp_solar_azimuth], x[BatchKey.gsp_solar_elevation]), 
                dim=1
            ).float()
            sun_out = self.sun_fc1(sun)
            out = torch.cat((out, sun_out), dim=1)
        
        # Fully connected layers.
        out = F.relu(self.fc3(out))
        out = self.fc4(out)

        out = out.reshape(batch_size, self.forecast_len)
        breakpoint()
        return out
    
    
if __name__=="__main__":
    
    history = 60
    forecast = 30

    sun_in = torch.zeros((3, (history+forecast)//30+1))
    gsp_id = torch.zeros((3, 1))
    # Shape: batch_size, seq_length, channel, height, width
    sat_data = torch.zeros((3, history//5 +1 - 3, 11, 24, 24))
    # shape: batch_size, seq_len, n_chans, height, width
    nwp_data = torch.zeros((3, (history+forecast)//60+1, 9, 12, 12))
    gsp = torch.zeros((3, (history+forecast)//30+1))

    batch = {
        BatchKey.gsp_solar_azimuth: sun_in,
        BatchKey.gsp_solar_elevation: sun_in,
        BatchKey.gsp_id: gsp_id,
        # Shape: batch_size, seq_length, channel, height, width
        BatchKey.satellite_actual: sat_data,
        BatchKey.nwp: nwp_data,
        BatchKey.gsp: gsp,
    }


    model = Model(
        include_gsp_yield_history = True,
        include_sun = True,
        include_nwp = True,
        embedding_dim = True,

        forecast_minutes = forecast,
        history_minutes = history,
        number_of_conv3d_layers = 4,
        conv3d_channels = 32,
        image_size_pixels = 24,
        nwp_image_size_pixels_height = 12,
        number_sat_channels = 11,
        number_nwp_channels = 9,
        fc1_output_features = 128,
        fc2_output_features = 128,
        fc3_output_features = 64,
    )

    model(batch)