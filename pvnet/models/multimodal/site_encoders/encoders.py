"""Encoder modules for the site-level PV data.

"""

import torch
from torch import nn

from ocf_datapipes.utils.consts import BatchKey

from pvnet.models.multimodal.linear_networks.networks import ResFCNet2
from pvnet.models.multimodal.site_encoders.basic_blocks import AbstractPVSitesEncoder


class SimpleLearnedAggregator(AbstractPVSitesEncoder):
    """A simple model which learns a different weighted-average across all of the PV sites for each
    GSP.
    
    Each sequence from each site is independently encodeded through some dense layers wih skip-
    connections, then the encoded form of each sequence is aggregated through a learned weighted-sum
    and finally put through more dense layers.
    
    This model was written to be a simplified version of a single-headed attention layer.
    """

    def __init__(
        self,
        sequence_length: int,
        num_sites: int,
        out_features: int,
        value_dim: int = 10,
        value_enc_resblocks: int = 2,
        final_resblocks: int = 2,
    ):
        """A simple sequence encoder and weighted-average model.
        
        Args:
            sequence_length: The time sequence length of the data.
            num_sites: Number of PV sites in the input data.
            out_features: Number of output features.
            value_dim: The number of features in each encoded sequence. Similar to the value 
                dimension in single- or multi-head attention.
            value_dim: The number of features in each encoded sequence. Similar to the value 
                dimension in single- or multi-head attention.
            value_enc_resblocks: Number of residual blocks in the value-encoder sub-network.
            final_resblocks: Number of residual blocks in the final sub-network.
        """

        super().__init__(sequence_length, num_sites, out_features)
        
        self.sequence_length = sequence_length
        self.num_sites = num_sites
        
        # Network used to encode each PV site sequence
        self._value_encoder = nn.Sequential(
            ResFCNet2(
                in_features=sequence_length,
                out_features=value_dim,
                fc_hidden_features=value_dim,
                n_res_blocks=value_enc_resblocks,
                res_block_layers=2,
                dropout_frac=0,
            ),
        )
        
        # The learned weighted average is stored in an embedding layer for ease of use
        self._attention_network = nn.Sequential(
            nn.Embedding(318, num_sites),
            nn.Softmax(dim=1),
        )
        
        # Network used to process weighted average
        self.output_network = ResFCNet2(
                in_features=value_dim,
                out_features=out_features,
                fc_hidden_features=value_dim,
                n_res_blocks=final_resblocks,
                res_block_layers=2,
                dropout_frac=0,
        )
        
    def _calculate_attention(self, x):
        gsp_ids = x[BatchKey.gsp_id].squeeze().int()
        attention = self._attention_network(gsp_ids)
        return attention
    
    def _encode_value(self, x):
        # Shape: [batch size, sequence length, PV site]
        pv_site_seqs = x[BatchKey.pv].float()
        batch_size = pv_site_seqs.shape[0]
        
        pv_site_seqs = pv_site_seqs.swapaxes(1,2).flatten(0,1)
        
        x_seq_enc = self._value_encoder(pv_site_seqs)
        x_seq_out = x_seq_enc.unflatten(0, (batch_size, self.num_sites))
        return x_seq_out        
        
    def forward(self, x):
        """Run model forward"""
        # Output has shape: [batch size, num_sites, value_dim]
        encodeded_seqs = self._encode_value(x)
        
        # Calculate learned averaging weights
        attn_avg_weights = self._calculate_attention(x)
        
        # Take weighted average across num_sites
        value_weighted_avg = (encodeded_seqs*attn_avg_weights.unsqueeze(-1)).sum(dim=1)
        
        # Put through final processing layers
        x_out = self.output_network(value_weighted_avg)
        
        return x_out
