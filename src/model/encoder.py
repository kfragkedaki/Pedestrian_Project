import torch
import numpy as np
from torch import nn
import math


class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input, **kwargs):
        return input + self.module(input, **kwargs)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        input_dim: int,
        embed_dim: int,
        val_dim: int = None,
        key_dim: int = None,
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // num_heads
        if key_dim is None:
            key_dim = val_dim

        self.num_heads = num_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(num_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(num_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(num_heads, input_dim, val_dim))

        self.W_out = nn.Parameter(torch.Tensor(num_heads, val_dim, embed_dim))
        self.attention_weights = None

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.num_heads, batch_size, graph_size, -1)
        shp_q = (self.num_heads, batch_size, n_query, -1)

        # Calculate queries, (num_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (num_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (num_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(
                compatibility
            )
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)
        self.attention_weights = attn.clone()

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3)
            .contiguous()
            .view(-1, self.num_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim),
        ).view(batch_size, n_query, self.embed_dim)

        return out

    def get_attention_weights(self):
        return self.attention_weights


class Normalization(nn.Module):
    def __init__(self, embed_dim, normalization="batch"):
        super(Normalization, self).__init__()

        normalizer_class = {"batch": nn.BatchNorm1d, "instance": nn.InstanceNorm1d}.get(
            normalization, None
        )

        self.normalizer = normalizer_class(embed_dim, affine=True)

    def init_parameters(self):
        for name, param in self.named_parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(
                *input.size()
            )  # same self.normalizer(input.permute(0,2,1)).permute(0,2,1)
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class MultiHeadAttentionLayer(nn.Sequential):
    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        feed_forward_hidden: int,
        normalization: str = "batch",
    ):
        super(MultiHeadAttentionLayer, self).__init__()

        self.attention = SkipConnection(
            MultiHeadAttention(num_heads, input_dim=embed_dim, embed_dim=embed_dim)
        )
        self.bn1 = Normalization(embed_dim, normalization)
        self.bn2 = Normalization(embed_dim, normalization)
        self.ff = SkipConnection(
            nn.Sequential(
                nn.Linear(embed_dim, feed_forward_hidden),
                nn.ReLU(),
                nn.Linear(feed_forward_hidden, embed_dim),
            )
            if feed_forward_hidden > 0
            else nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, input, mask=None):
        # Pass mask to the MultiHeadAttention
        out = self.attention(input, mask=mask)
        out = self.bn1(out)
        out = self.ff(out)
        out = self.bn2(out)

        return out


class GraphAttentionEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_attention_layers: int = 3,
        num_heads: int = 8,
        feed_forward_hidden: int = 512,
        normalization: str = "batch",
    ):
        super(GraphAttentionEncoder, self).__init__()

        self.layers = nn.ModuleList(
            [
                MultiHeadAttentionLayer(
                    num_heads, embed_dim, feed_forward_hidden, normalization
                )
                for _ in range(num_attention_layers)
            ]
        )

    def forward(self, x, mask=None):
        # Batch multiply to get initial embeddings of nodes
        for layer in self.layers:
            out = layer(x, mask=mask)

        return out  # (batch_size, graph_size, embed_dim)
