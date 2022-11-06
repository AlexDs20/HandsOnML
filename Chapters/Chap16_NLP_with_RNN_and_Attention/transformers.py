from typing import Optional
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dk: Optional=None, dv: Optional=None, mask: Optional=None):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.dk = dk if dk is not None else d_model // heads
        self.dv = dv if dv is not None else d_model // heads
        self.heads = heads
        self.mask = mask

        assert d_model % self.dk == 0
        assert d_model % self.dv == 0

        self.WQ = nn.Linear(self.d_model, self.heads * self.dk, bias = False)
        self.WK = nn.Linear(self.d_model, self.heads * self.dk, bias = False)
        self.WV = nn.Linear(self.d_model, self.heads * self.dv, bias = False)

        self.WO = nn.Linear(self.heads * self.dv, self.d_model)

    def forward(self, Q, K, V):
        N = Q.shape[0]
        vl, kl, ql = V.shape[1], K.shape[1], Q.shape[1]

        assert kl == vl

        Q = self.WQ(Q).reshape(N, ql, self.heads, self.dk)
        K = self.WK(K).reshape(N, kl, self.heads, self.dk)
        V = self.WV(V).reshape(N, vl, self.heads, self.dv)

        # Do QK^T  -> [N, self.heads, ql, kl]
        compat = torch.einsum('nqhk,nlhk->nhql', Q, K) / self.dk**(1/2)

        # Masking when doing QQT in decoder
        if self.mask is not None:    # mask upper right triangle, not diagonal
            # index for upper right triangle excluding diagonal
            indices = torch.triu_indices(ql, kl, offset=1)
            compat[..., indices[0], indices[1]] = float('-inf')

        weights = torch.softmax(compat, dim=3)

        # Do weights*V -> [N,ql,self.heads, self.dv]
        # + reshape
        attention = torch.einsum('nhqk,nkhv->nqhv', weights, V).reshape(
            N, ql, self.heads * self.dv
        )

        out = self.WO(attention)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, d_model=512, heads=8, dropout=0.1, d_ff=2048):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, heads)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention = self.attention(x, x, x)

        x = self.norm1(self.dropout(attention) + x)
        ff = self.ff(x)
        out = self.norm2(self.dropout(ff) + x)
        return out


class PositionEmbedding(nn.Module):
    def __init__(self, max_length, d_model):
        super(PositionEmbedding, self).__init__()
        if max_length % 2 == 1: max_length += 1  # must be even
        p, i = torch.meshgrid(torch.arange(max_length), torch.arange(d_model // 2), indexing='ij')

        P = torch.empty(1, max_length, d_model)
        P[0, :, ::2] = torch.sin(p / 10000**(2 * i / max_length))
        P[0, :, 1::2] = torch.cos(p / 10000**(2 * i / max_length))

        self.P = P

    def forward(self, x):
        shape = x.shape
        return self.P[:, :shape[-2], :shape[-1]]


class Transformer(nn.Module):
    def __init__(self,
                 d_model=512,
                 dropout=0.1,
                 en_vocab=1000,
                 en_layers=6,
                 en_heads=8,
                 en_d_ff=512*4,
                 en_max_length=100):
        super(Transformer, self).__init__()

        self.en_embedding = nn.Embedding(en_vocab, d_model)
        self.en_pos_embedding = PositionEmbedding(en_max_length, d_model)
        self.en_emb_dropout = nn.Dropout(dropout)

        self.encoder = nn.ModuleList(
            [
                EncoderBlock(d_model, en_heads, dropout, en_d_ff)
                for _ in range(en_layers)
            ]
        )

    def forward(self, x):
        # Encoder part
        x = self.en_embedding(x)
        x = x + self.en_pos_embedding(x)

        for encoder_layer in self.encoder:
            x = encoder_layer(x)

        return x
        # Decoder part


if __name__ == '__main__':
    # Test positional Embedding
    input = torch.randint(10, (4, 20))
    pe = PositionEmbedding(30, 100)
    print(pe(input).shape)

    # Test Encoder block
    input = torch.rand((4, 20, 512))
    eb = EncoderBlock()
    print(eb(input).shape)

    # Test transformer
    input = torch.randint(10, (4, 20))
    tf = Transformer()
    print(tf(input).shape)
