import torch
import torch.nn as nn
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

    def forward(self, x):
        new_hidden = self.i2h(x) + self.h2h(hidden)
        out = self.i2o(x) + self.h2o(hidden)
        return out, new_hidden


class LinearModel(nn.Module):
    def __init__(self, in_features, out_features, *args, **kwargs):
        super(LinearModel, self).__init__(*args, **kwargs)
        self.i2o = nn.Linear(in_features, out_features)

    def forward(self, x):
        out = self.i2o(x.squeeze())
        return out


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


class RNN(nn.Module):
    def __init__(self, in_features, hidden_features, num_layers, out_features, *args, **kwargs):
        super(RNN, self).__init__(*args, **kwargs)
        self.rnn = nn.RNN(in_features, hidden_features, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.linear(out)
        return out

def plot_batch(batch):
    batch = batch.cpu().numpy()
    for b in batch:
        bt = b.T
        for feature in bt:
            plt.plot(feature)


#------------------------------
epochs = 15
bs = 512
features = 1
length = 200
order = 20
samples = 20*bs
hidden_features = 128
out_features = features
num_layers = 3
lr = 1e-3
#------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model = RecurrentNeuron(in_features=features, hidden_features=hidden_features, out_features=features).to(device)
#model = RNN(features, hidden_features, num_layers, out_features).to(device)
model = LinearModel(length-1, out_features).to(device)

ds = TSDataSet(features=features, order=order, length=length, samples=samples)
dl = DataLoader(ds, batch_size=bs, num_workers=8, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.MSELoss()

# Training
for epoch in range(epochs):
    if epoch+1 % 2 == 0:
        for g in optimizer.param_groups:
            g['lr'] /= 2

    for batch in dl:
        batch = batch.to(device)
        data = batch[:, :-1, :]
        target = batch[:, -1, :]

        out = model(data)
        loss = criterion(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(epoch, loss.item())


# Testing and visualise the model
bs = 1
samples = 1
ds = TSDataSet(features=features, order=order, length=length, samples=samples)
dl = DataLoader(ds, batch_size=bs, num_workers=8, shuffle=False)

with torch.no_grad():
    predictions = torch.zeros((samples, length, features))
    truth = torch.zeros((samples, length, features))
    for batch in dl:
        batch = batch.to(device)
        predictions[:bs, :-1, :] = batch[:, :-1, :]
        truth[:bs] = batch

        out = model(batch[:, :-1, :])
        predictions[:bs, -1, :] = out


fig, ax = plt.subplots()
plot_batch(truth)
plot_batch(predictions)
plt.show()
