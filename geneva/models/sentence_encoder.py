# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""A sentence encoder that trains a GRU on top a sequence
of Glove word embedding"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class SentenceEncoder(nn.Module):
    """A sentence encoder. Takes input list of Glove word
    embedding, a GRU is trained on top, the final hidden state
    is used as the sentence encoding."""
    def __init__(self, cfg):
        super(SentenceEncoder, self).__init__()
        self.gru = nn.GRU(300,  # input size
                          cfg.embedding_dim // 2,   # hidden size
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True)

        self.layer_norm = nn.LayerNorm(cfg.embedding_dim)
        self.cfg = cfg

    def forward(self, words, lengths):
        lengths = lengths.long()
        reorder = False
        sorted_len, indices = torch.sort(lengths, descending=True)
        if not torch.equal(sorted_len, lengths):
            _, reverse_sorting = torch.sort(indices)
            reorder = True
            words = words[indices]
            lengths = lengths[indices]

        lengths[lengths == 0] = 1

        packed_padded_sequence = pack_padded_sequence(words,
                                                      lengths,
                                                      batch_first=True)

        self.gru.flatten_parameters()
        _, h = self.gru(packed_padded_sequence)
        h = h.permute(1, 0, 2).contiguous()
        h = h.view(-1, self.cfg.embedding_dim)

        if reorder:
            h = h[reverse_sorting]

        h = self.layer_norm(h)

        return h


class SentenceEncoderSA(nn.Module):
    """
    Modified sentence encoder. Instead of GRU, we want to use the self-attention layer
    to encode the sentences to better address the mis-shaped or mis-colored objects.
    Train on the list of Glove word embedding.
    Output layer is used as the sentence encoding.
    """

    def __init__(self, cfg):
        super(SentenceEncoderSA, self).__init__()
        self.transformer = TransformerLayer(
            300, cfg.embedding_dim // 2, num_layers=1, batch_first=True, bidirectional=True)

        self.layer_norm = nn.LayerNorm(cfg.embedding_dim)
        self.cfg = cfg

    def forward(self, words, lengths):
        lengths = lengths.long()
        reorder = False
        sorted_len, indices = torch.sort(lengths, descending=True)
        if not torch.equal(sorted_len, lengths):
            _, reverse_sorting = torch.sort(indices)
            reorder = True
            words = words[indices]
            lengths = lengths[indices]

        lengths[lengths == 0] = 1

        packed_padded_sequence = pack_padded_sequence(words,
                                                      lengths,
                                                      batch_first=True)

        self.gru.flatten_parameters()
        _, h = self.gru(packed_padded_sequence)
        h = h.permute(1, 0, 2).contiguous()
        h = h.view(-1, self.cfg.embedding_dim)

        if reorder:
            h = h[reverse_sorting]

        h = self.layer_norm(h)

        return h


class LearnedPositionEncoding(nn.Embedding):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        x = x + weight[: x.size(0), :]
        return self.dropout(x)


class TransformerLayer(nn.Module):

    def __init__(self, vocab_size, position_enc, d_model=1024, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1):
        super(TransformerLayer, self).__init__()

        # Preprocess
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.pos_encoder_src = position_enc(d_model=512)
        # tgt
        self.pos_encoder_tgt = position_enc(d_model=512)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model,nhead,dim_feedforward,dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer,num_encoder_layers,encoder_norm)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model,nhead,dim_feedforward,dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer,num_decoder_layers,decoder_norm)
        self.output_layer = nn.Linear(d_model,vocab_size)

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead


    def forward(self, src,tgt,src_mask = None,tgt_mask = None,
                memory_mask = None,src_key_padding_mask = None,
                tgt_key_padding_mask = None,memory_key_padding_mask = None):

        # word embedding
        src = self.embedding(src)
        tgt = self.embedding(tgt)

        # shape check
        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")
        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        # position encoding
        src = self.pos_encoder_src(src)
        tgt = self.pos_encoder_tgt(tgt)

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        output = self.output_layer(output)
        # return output
        return softmax(output,dim = 2)


    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)