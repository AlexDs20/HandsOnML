import torch
from torch.utils.data import Dataset, DataLoader

class RecurrentNeuron(torch.nn.Module):
    def __init__(self):
        super(RecurrentNeuron, self).__init__()
        self.neuron = torch.nn.Parameter(torch.randn())

    def forward(self, x):
        out = x
        return out


class TSDataSet(Dataset):
    def __init__(self):
        super(TSDataSet, self).__init__()
        self.len = 100
        self.order = 20

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        samples = 10000
        x = torch.arange(0, samples).repeat(self.order, 1)

        A = torch.randn((self.order, 1))
        n = torch.randint(0, self.order, (self.order, 1))
        p = 2*torch.pi*torch.randn((self.order, 1))

        out = A * torch.cos((2*torch.pi)/samples * n * x)
        return out.sum(dim=0)

ds = TSDataSet()
import matplotlib.pyplot as plt
TS = next(iter(ds))
fig, ax = plt.subplots()
plt.plot(TS)
plt.show()
