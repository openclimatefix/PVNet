"""Basic layers for composite models"""


import torch
from torch import nn


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
        return torch.cat((x, emb), dim=1)
