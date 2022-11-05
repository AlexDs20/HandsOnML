from typing import Optional
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dk: Optional=None, dv: Optional=None):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.dk = dk if dk is not None else d_model // heads
        self.dv = dv if dv is not None else d_model // heads
        self.heads = heads

        assert d_model % dk == 0
        assert d_model % dv == 0

        self.WQ = nn.Linear(d_model, heads * dk)
        self.WK = nn.Linear(d_model, heads * dk)
        self.WV = nn.Linear(d_model, heads * dv)

        self.WO = nn.Linear(heads * dv, d_model)

    def forward(self, Q, K, V):
        N = Q.shape[0]
        vl, kl, ql = V.shape[1], K.shape[1], Q.shape[1]

        assert kl == vl

        Q = self.WQ(Q).reshape(N, ql, self.heads, self.dk)
        K = self.WK(K).reshape(N, kl, self.heads, self.dk)
        V = self.WV(V).reshape(N, vl, self.heads, self.dv)

        # Do QK^T  -> [N, self.heads, ql, kl]
        compat = torch.einsum('nqhk,nlhk->nhql', Q, K)

        weights = torch.softmax(compat / torch.sqrt(self.dk), dim=3)

        # Do weights*V -> [N,ql,self.heads, self.dv]
        # + reshape
        attention = torch.einsum('nhqk,nkhv->nqhv', weights, V).reshape(
            N, ql, self.heads * self.dv
        )
