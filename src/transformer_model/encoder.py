from typing import Optional
import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import (
    MultiheadAttention,
    Linear,
    Dropout,
    BatchNorm1d,
    TransformerEncoderLayer,
)


def model_factory(config, data):
    task = config["task"]
    feat_dim = data.feature_df.shape[1]  # dimensionality of data features
    max_seq_len = data.max_seq_len
    # TODO Claire

    if task == "imputation":
        return TSTransformerEncoder(
            feat_dim,
            max_seq_len,
            config["embedding_dim"],
            config["num_heads"],
            config["num_layers"],
            config["hidden_dim"],
            dropout=config["dropout"],
            pos_encoding=config["pos_encoding"],
            activation=config["activation"],
            norm=config["normalization_layer"],
        )
    else:
        raise ValueError("Model class for task '{}' does not exist".format(task))


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError(
        "pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding)
    )


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/embedding_dim))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/embedding_dim))
        \text{where pos is the word position and i is the embed idx)
    Args:
        embedding_dim: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, embedding_dim, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embedding_dim)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(10000.0) / embedding_dim)
        )  # (embedding_dim/2,)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(1).transpose(
            0, 1
        )  # (1, max_len, embedding_dim)
        self.register_buffer(
            "pe", pe
        )  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, embedding_dim, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(
            torch.empty(1, max_len, embedding_dim)  # (1, max_len, embedding_dim)
        )  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        embedding_dim: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        hidden_dim: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(
        self, embedding_dim, nhead, hidden_dim=2048, dropout=0.1, activation="relu"
    ):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            embedding_dim, nhead, dropout=dropout, batch_first=True
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(embedding_dim, hidden_dim)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(hidden_dim, embedding_dim)

        self.norm1 = BatchNorm1d(
            embedding_dim, eps=1e-5
        )  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(embedding_dim, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        #  Transformer Model and Residual connection
        src2 = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[
            0
        ]  # attn_output, attn_output_weights
        src = src + self.dropout1(src2)  # (batch_size, seq_len, embedding_dim)

        # Normalization
        src = src.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
        src = self.norm1(src)
        src = src.permute(0, 2, 1)  # restore (batch_size, seq_len, embedding_dim)

        # Feed Forward Layer and Residual connection
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (batch_size, seq_len, embedding_dim)

        # Normalization
        src = src.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
        src = self.norm2(src)
        src = src.permute(0, 2, 1)  # restore (batch_size, seq_len, embedding_dim)
        return src


class TSTransformerEncoder(nn.Module):

    def __init__(
        self,
        feat_dim,
        max_len,
        embedding_dim,
        n_heads,
        num_layers,
        hidden_dim,
        dropout=0.1,
        pos_encoding="fixed",
        activation="gelu",
        norm="BatchNorm",
        freeze=False,
    ):
        super(TSTransformerEncoder, self).__init__()

        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, embedding_dim)
        self.pos_enc = get_pos_encoder(pos_encoding)(
            embedding_dim, dropout=dropout * (1.0 - freeze), max_len=max_len
        )

        if norm == "LayerNorm":
            encoder_layer = TransformerEncoderLayer(
                embedding_dim,
                self.n_heads,
                hidden_dim,
                dropout * (1.0 - freeze),
                activation=activation,
            )
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(
                embedding_dim,
                self.n_heads,
                hidden_dim,
                dropout * (1.0 - freeze),
                activation=activation,
            )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.act = _get_activation_fn(activation)
        self.dropout1 = nn.Dropout(dropout)
        self.output_layer = nn.Linear(embedding_dim, feat_dim)
        self.feat_dim = feat_dim

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = self.project_inp(X) * math.sqrt(
            self.embedding_dim
        )  # [batch_size, seq_length, embedding_dim] project input vectors to embedding_dim dimensional space
        inp = self.pos_enc(inp)  # add positional encoding

        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        #  padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        output = self.transformer_encoder(
            inp, src_key_padding_mask=~padding_masks
        )  # (batch_size, seq_len, embedding_dim) # x3
        embeddings_original = output
        output = self.act(
            output
        )  # the output transformer encoder/decoder embeddings don't include non-linearity
        embeddings = output

        output = self.dropout1(output)  # (batch_size, seq_length, embedding_dim)

        # Most probably defining a Linear(embedding_dim,feat_dim) vectorizes the operation over (seq_length, batch_size).
        output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)

        return output, (embeddings, embeddings_original)
