import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from typing import NamedTuple

from .graph_encoder import GraphAttentionEncoder
from src.graph.evrp_network import EVRPNetwork
from src.graph.evrp_graph import EVRPGraph
from src.utils.beam_search import CachedLookup

import os


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """

    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key],
        )


class AttentionPredictionModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        dataset=None,
        n_encode_layers: int = 2,
        normalization: str = "batch",
        n_heads: int = 8,
        checkpoint_encoder: bool = False,
        opts: dict = None,
    ):
        super(AttentionPredictionModel, self).__init__()

        self.opts = opts

        self.dataset = dataset
        self.n_heads = n_heads
        self.n_encode_layers = n_encode_layers
        self.checkpoint_encoder = checkpoint_encoder
        self.encoder_data = dict()

        self._initialize_input(embedding_dim)

        self.encoder = GraphAttentionEncoder(
            num_heads=n_heads,
            embed_dim=embedding_dim,
            num_attention_layers=n_encode_layers,
            normalization=normalization,
        )

    def forward(
        self,
        input: dict = {},
        padding_value: int = 1000,
    ):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """
        mask_padded_input = [input[key] == padding_value for key in input.keys()] # batch_size, graph_size, graph_size

        mask_non_neighbors = (
            input["coords"][:, :, None, :] - input["coords"][:, None, :, :]
        ).norm(
            p=2, dim=-1
        ) > self.opts.battery_limit  # batch_size, chunk_size, parameters

        if (
            self.checkpoint_encoder and self.training
        ):  # Only checkpoint if we need gradients
            embeddings = checkpoint(
                self.encoder, self._init_embed(input), mask_padded_input
            )
        else:
            embeddings = self.encoder(self._init_embed(input), mask=mask_padded_input)

        self.encoder_data["input"] = input["coords"].detach().cpu()
        self.encoder_data["embeddings"] = embeddings.detach().cpu()

        return self.encoder_data

    def _init_embed(self, input):
        return self.init_embed_node(
            torch.cat((input["coords"], *(input[feat] for feat in self.features)), -1)
        )

    def _initialize_input(self, embedding_dim: int):
        self.node_dim = 6  # x, y, v_x, v_y, a_x, a_y
        self.features = ("velocity", "acceleration")

        # To map input to embedding space
        self.init_embed_node = nn.Linear(self.node_dim, embedding_dim)
