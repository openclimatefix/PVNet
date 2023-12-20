"""Encoder modules for the site-level PV data.

"""

import torch
from ocf_datapipes.batch import BatchKey
from torch import nn

from pvnet.models.multimodal.linear_networks.networks import ResFCNet2
from pvnet.models.multimodal.site_encoders.basic_blocks import AbstractPVSitesEncoder


class SimpleLearnedAggregator(AbstractPVSitesEncoder):
    """A simple model which learns a different weighted-average across all PV sites for each GSP.

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

        # The learned weighted average is stored in an embedding layer for ease of use
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

        pv_site_seqs = pv_site_seqs.swapaxes(1, 2).flatten(0, 1)

        x_seq_enc = self._value_encoder(pv_site_seqs)
        x_seq_out = x_seq_enc.unflatten(0, (batch_size, self.num_sites))
        return x_seq_out

    def forward(self, x):
        """Run model forward"""
        # Output has shape: [batch size, num_sites, value_dim]
        encodeded_seqs = self._encode_value(x)

        # Calculate learned averaging weights
        attn_avg_weights = self._calculate_attention(x)

        # Take weighted average across num_sites
        value_weighted_avg = (encodeded_seqs * attn_avg_weights.unsqueeze(-1)).sum(dim=1)

        # Put through final processing layers
        x_out = self.output_network(value_weighted_avg)

        return x_out


class SingleAttentionNetwork(AbstractPVSitesEncoder):
    """A simple attention-based model with a single multihead attention layer

    For the attention layer the query is based on the target GSP alone, the key is based on the PV
    ID and the recent PV data, the value is based on the recent PV data.

    """

    def __init__(
        self,
        sequence_length: int,
        num_sites: int,
        out_features: int,
        kdim: int = 10,
        pv_id_embed_dim: int = 10,
        num_heads: int = 2,
        n_kv_res_blocks: int = 2,
        kv_res_block_layers: int = 2,
        use_pv_id_in_value: bool = False,
    ):
        """A simple attention-based model with a single multihead attention layer

        Args:
            sequence_length: The time sequence length of the data.
            num_sites: Number of PV sites in the input data.
            out_features: Number of output features. In this network this is also the embed and
                value dimension in the multi-head attention layer.
            kdim: The dimensions used the keys.
            pv_id_embed_dim: Number of dimensiosn used in the PD ID embedding layer(s).
            num_heads: Number of parallel attention heads. Note that `out_features` will be split
                across `num_heads` so `out_features` must be a multiple of `num_heads`.
            n_kv_res_blocks: Number of residual blocks to use in the key and value encoders.
            kv_res_block_layers: Number of fully-connected layers used in each residual block within
                the key and value encoders.
            use_pv_id_in_value: Whether to use a PV ID embedding in network used to produce the
                value for the attention layer.

        """
        super().__init__(sequence_length, num_sites, out_features)

        self.gsp_id_embedding = nn.Embedding(318, out_features)
        self.pv_id_embedding = nn.Embedding(num_sites, pv_id_embed_dim)
        self._pv_ids = nn.parameter.Parameter(torch.arange(num_sites), requires_grad=False)
        self.use_pv_id_in_value = use_pv_id_in_value

        if use_pv_id_in_value:
            self.value_pv_id_embedding = nn.Embedding(num_sites, pv_id_embed_dim)

        self._value_encoder = nn.Sequential(
            ResFCNet2(
                in_features=sequence_length + int(use_pv_id_in_value) * pv_id_embed_dim,
                out_features=out_features,
                fc_hidden_features=sequence_length,
                n_res_blocks=n_kv_res_blocks,
                res_block_layers=kv_res_block_layers,
                dropout_frac=0,
            ),
        )

        self._key_encoder = nn.Sequential(
            ResFCNet2(
                in_features=sequence_length + pv_id_embed_dim,
                out_features=kdim,
                fc_hidden_features=pv_id_embed_dim + sequence_length,
                n_res_blocks=n_kv_res_blocks,
                res_block_layers=kv_res_block_layers,
                dropout_frac=0,
            ),
        )

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=out_features,
            kdim=kdim,
            vdim=out_features,
            num_heads=num_heads,
            batch_first=True,
        )

    def _encode_query(self, x):
        gsp_ids = x[BatchKey.gsp_id].squeeze().int()
        query = self.gsp_id_embedding(gsp_ids).unsqueeze(1)
        return query

    def _encode_key(self, x):
        # Shape: [batch size, sequence length, PV site]
        pv_site_seqs = x[BatchKey.pv].float()
        batch_size = pv_site_seqs.shape[0]

        # PV ID embeddings are the same for each sample
        pv_id_embed = torch.tile(self.pv_id_embedding(self._pv_ids), (batch_size, 1, 1))

        # Each concated (PV sequence, PV ID embedding) is processed with encoder
        x_seq_in = torch.cat((pv_site_seqs.swapaxes(1, 2), pv_id_embed), dim=2).flatten(0, 1)
        key = self._key_encoder(x_seq_in)

        # Reshape to [batch size, PV site, kdim]
        key = key.unflatten(0, (batch_size, self.num_sites))
        return key

    def _encode_value(self, x):
        # Shape: [batch size, sequence length, PV site]
        pv_site_seqs = x[BatchKey.pv].float()
        batch_size = pv_site_seqs.shape[0]

        if self.use_pv_id_in_value:
            # PV ID embeddings are the same for each sample
            pv_id_embed = torch.tile(self.value_pv_id_embedding(self._pv_ids), (batch_size, 1, 1))
            # Each concated (PV sequence, PV ID embedding) is processed with encoder
            x_seq_in = torch.cat((pv_site_seqs.swapaxes(1, 2), pv_id_embed), dim=2).flatten(0, 1)
        else:
            # Encode each PV sequence independently
            x_seq_in = pv_site_seqs.swapaxes(1, 2).flatten(0, 1)

        value = self._value_encoder(x_seq_in)

        # Reshape to [batch size, PV site, vdim]
        value = value.unflatten(0, (batch_size, self.num_sites))
        return value

    def _attention_forward(self, x, average_attn_weights=True):
        query = self._encode_query(x)
        key = self._encode_key(x)
        value = self._encode_value(x)

        attn_output, attn_weights = self.multihead_attn(
            query, key, value, average_attn_weights=average_attn_weights
        )

        return attn_output, attn_weights

    def forward(self, x):
        """Run model forward"""
        attn_output, attn_output_weights = self._attention_forward(x)

        # Reshape from [batch_size, 1, vdim] to [batch_size, vdim]
        x_out = attn_output.squeeze()

        return x_out
