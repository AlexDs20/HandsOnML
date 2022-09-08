import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class RecurrentNeuron(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(RecurrentNeuron, self).__init__()
        self.hidden_features = hidden_features

        self.i2h = nn.Linear(in_features, hidden_features)
        self.h2h = nn.Linear(hidden_features, hidden_features)

    def forward(self, x, hidden):
        out = F.relu(self.i2h(x) + self.h2h(hidden))
        return out


class TSDataSet(Dataset):
    def __init__(self, features=2, order=10, len=100, samples=10000):
        super(TSDataSet, self).__init__()

        self.len      = len
        self.order    = order
        self.features = features
        self.samples  = samples

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = torch.arange(0, self.samples).repeat(self.features, self.order, 1)

        A = torch.randn((self.features, self.order, 1))
        n = torch.randint(0, self.order, (self.features, self.order, 1))
        p = 2*torch.pi*torch.randn((self.features, self.order, 1))

        out = A * torch.cos((2*torch.pi)/self.samples * n * x)
        out = out.sum(dim=1)
        return out


bs = 5
features = 2
hidden_features = 128
model = RecurrentNeuron(in_features=features, hidden_features=hidden_features)

ds = TSDataSet(len=10)
dl = DataLoader(ds, batch_size=bs)

hidden = torch.zeros((bs, hidden_features))
for idx, batch in enumerate(dl):
    print(idx)
    for t in range(batch.shape[-1]):
        # data shape - (N, C)
        data = batch[:, :, t]
        hidden = model(data, hidden)
