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


class DecoderBlock(nn.Module):
    def __init__(self, d_model=512, heads=8, dropout=0.1, d_ff=2048):
        super(DecoderBlock, self).__init__()
        self.masked_attention = MultiHeadAttention(d_model, heads, mask=True)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attention = MultiHeadAttention(d_model, heads)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, outputs, K, V):
        x = self.masked_attention(outputs, outputs, outputs)
        Q = self.norm1(x + self.dropout(x))
        x = self.cross_attention(Q, K, V)
        x = self.norm2(Q + self.dropout(x))
        ff = self.ff(x)
        x = self.norm3(x + self.dropout(ff))
        return x


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
                 en_max_length=100,
                 de_vocab=1000,
                 de_layers=6,
                 de_heads=8,
                 de_d_ff=512*4,
                 de_max_length=100
                 ):
        super(Transformer, self).__init__()

        self.en_embedding = nn.Embedding(en_vocab, d_model)
        self.pos_embedding = PositionEmbedding(max(en_max_length, de_max_length), d_model)
        self.dropout = nn.Dropout(dropout)

        self.encoder = nn.ModuleList(
            [
                EncoderBlock(d_model, en_heads, dropout, en_d_ff)
                for _ in range(en_layers)
            ]
        )

        self.de_embedding = nn.Embedding(de_vocab, d_model)
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(d_model, de_heads, dropout, de_d_ff)
                for _ in range(de_layers)
            ]
        )

        self.linear = nn.Linear(d_model, de_vocab)

    def forward(self, x, dec_in):
        # Encoder part
        x = self.en_embedding(x)
        x = self.dropout(x + self.pos_embedding(x))

        for encoder_layer in self.encoder:
            x = encoder_layer(x)

        # Decoder part
        dec_in = self.de_embedding(dec_in)
        dec_in = self.dropout(dec_in + self.pos_embedding(dec_in))

        for decoder_layer in self.decoder:
            dec_in = decoder_layer(dec_in, x, x)

        out = self.linear(dec_in)

        return out


if __name__ == '__main__':
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
    # Test positional Embedding
    input = torch.randint(10, (4, 20)).to(device)
    pe = PositionEmbedding(30, 100).to(device)
    print(pe(input).shape)

    # Test Encoder block
    input = torch.rand((4, 20, 512))
    eb = EncoderBlock()
    print(eb(input).shape)

    # Test Decoder block
    outputs = torch.rand(4, 20, 512)
    K = torch.rand((4, 20, 512))
    V = torch.rand((4, 20, 512))
    db = DecoderBlock()
    print(db(outputs, K, V).shape)

    # Test transformer
    input = torch.randint(10, (4, 20))
    dec = torch.randint(15, (4, 22))
    tf = Transformer()
    print(tf(input, dec).shape)

