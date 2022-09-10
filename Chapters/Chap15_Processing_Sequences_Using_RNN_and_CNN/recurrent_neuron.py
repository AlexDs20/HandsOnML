import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class RecurrentNeuron(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, *args, **kwargs):
        super(RecurrentNeuron, self).__init__(*args, **kwargs)
        self.hidden_features = hidden_features

        self.i2h = nn.Linear(in_features, hidden_features)
        self.h2h = nn.Linear(hidden_features, hidden_features)

        self.i2o = nn.Linear(in_features, out_features)
        self.h2o = nn.Linear(hidden_features, out_features)

    def forward(self, x, hidden):
        new_hidden = F.sigmoid(self.i2h(x) + self.h2h(hidden))
        out        = x + F.sigmoid(self.i2o(x) + self.h2o(hidden))
        return out, new_hidden


class TSDataSet(Dataset):
    def __init__(self, features, order, samples, length):
        super(TSDataSet, self).__init__()

        self.samples  = samples
        self.order    = order
        self.features = features
        self.length   = length

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        x = torch.arange(0, self.length).repeat(self.order, self.features, 1)
        n = torch.randint(1, self.order+1, (self.order, self.features, 1))
        A = torch.randn((self.order, self.features, 1))

        out = A * torch.cos(((2*torch.pi)/self.length) * n * x)
        out = torch.transpose(out.sum(dim=0), 0, 1)
        return out

def plot_batch(batch):
    batch = batch.cpu().numpy()
    for b in batch:
        bt = b.T
        for feature in bt:
            plt.plot(feature)


epochs = 10
bs = 64
features = 1
length = 3000
order = 3
samples = 320
hidden_features = 128
lr = 1e-3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RecurrentNeuron(in_features=features, hidden_features=hidden_features, out_features=features).to(device)

ds = TSDataSet(features=features, order=order, length=length, samples=samples)
dl = DataLoader(ds, batch_size=bs, num_workers=8, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.MSELoss()

for epoch in range(epochs):
    for batch in dl:
        hidden = torch.zeros((bs, hidden_features), device=device)
        batch = batch.to(device)
        loss = 0
        optimizer.zero_grad()

        for t in range(batch.shape[1]-1):
            data = batch[:, t, :]
            target = batch[:, t+1, :]
            out, hidden = model(data, hidden)
            l = criterion(out, target)
            loss = loss + l
        loss.backward()
        optimizer.step()
        print(epoch, loss.item())


# Testing the model
bs = 1
samples = 1
ds = TSDataSet(features=features, order=order, length=length, samples=samples)
dl = DataLoader(ds, batch_size=bs, num_workers=8, shuffle=False)

n, m = 1200, length
with torch.no_grad():
    predictions = torch.zeros((samples, m, features))
    truth = torch.zeros((samples, m, features))
    for batch in dl:
        batch = batch.to(device)
        predictions[:bs, :n, :] = batch[:, :n, :]
        truth[:bs,] = batch[:,]

        hidden = torch.zeros((bs, hidden_features), device=device)
        for t in range(n):    # I run the first n elements through the model to get the hidden state
            data = batch[:, t, :]
            out, hidden = model(data, hidden)

        for t in range(n, m):
            out, hidden = model(out, hidden)
            predictions[:bs, t, :] = out

fig, ax = plt.subplots()
plot_batch(truth)
plot_batch(predictions)
plt.show()
