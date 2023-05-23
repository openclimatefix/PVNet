"""Basic layers for composite models"""

import warnings

import torch
from torch import _VF, nn


class ImageEmbedding(nn.Module):
    """A embedding layer which concatenates an ID embedding as a new channel onto 3D inputs."""

    def __init__(self, num_embeddings, sequence_length, image_size_pixels, **kwargs):
        """A embedding layer which concatenates an ID embedding as a new channel onto 3D inputs.

        The embedding is a single 2D image and is appended at each step in the 1st dimension
        (assumed to be time).

        Args:
            num_embeddings: Size of the dictionary of embeddings
            sequence_length: The time sequence length of the data.
            image_size_pixels: The spatial size of the image. Assumed square.
            **kwargs: See `torch.nn.Embedding` for more possible arguments.
        """
        super().__init__()
        self.image_size_pixels = image_size_pixels
        self.sequence_length = sequence_length
        self._embed = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=image_size_pixels * image_size_pixels,
            **kwargs,
        )

    def forward(self, x, id):
        """Append ID embedding to image"""
        emb = self._embed(id)
        emb = emb.reshape((-1, 1, 1, self.image_size_pixels, self.image_size_pixels))
        emb = emb.repeat(1, 1, self.sequence_length, 1, 1)
        x = torch.cat((x, emb), dim=1)
        return x


class CompleteDropoutNd(nn.Module):
    """A layer used to completely drop out all elements of a N-dimensional sample.

    Each sample will be zeroed out independently on every forward call with probability `p` using
    samples from a Bernoulli distribution.

    """

    __constants__ = ["p", "inplace", "n_dim"]
    p: float
    inplace: bool
    n_dim: int

    def __init__(self, n_dim, p=0.5, inplace=False):
        """A layer used to completely drop out all elements of a N-dimensional sample.

        Args:
            n_dim: Number of dimensions of each sample not including channels. E.g. a sample with
                shape (channel, time, height, width) would use `n_dim=3`.
            p: probability of a channel to be zeroed. Default: 0.5
            training: apply dropout if is `True`. Default: `True`
            inplace: If set to `True`, will do this operation in-place. Default: `False`
        """
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p = p
        self.inplace = inplace
        self.n_dim = n_dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run dropout"""
        p = self.p
        inp_dim = input.dim()

        if inp_dim not in (self.n_dim + 1, self.n_dim + 2):
            warn_msg = (
                f"CompleteDropoutNd: Received a {inp_dim}-D input. Expected either a single sample"
                f" with {self.n_dim+1} dimensions, or a batch of samples with {self.n_dim+2}"
                " dimensions."
            )
            warnings.warn(warn_msg)

        is_batched = inp_dim == self.n_dim + 2
        if not is_batched:
            input = input.unsqueeze_(0) if self.inplace else input.unsqueeze(0)

        input = input.unsqueeze_(1) if self.inplace else input.unsqueeze(1)

        result = (
            _VF.feature_dropout_(input, p, self.training)
            if self.inplace
            else _VF.feature_dropout(input, p, self.training)
        )

        result = result.squeeze_(1) if self.inplace else result.squeeze(1)

        if not is_batched:
            result = result.squeeze_(0) if self.inplace else result.squeeze(0)

        return result
