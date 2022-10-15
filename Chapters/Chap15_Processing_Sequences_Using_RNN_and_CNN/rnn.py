import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

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
        A = torch.randn((self.order, self.features, 1))/self.order

        out = A * torch.cos(((2*torch.pi)/self.length) * n * x)
        out = torch.transpose(out.sum(dim=0), 0, 1)
        return out

def plot_batch(batch):
    batch = batch.cpu().numpy()
    for b in batch:
        bt = b.T
        for feature in bt:
            plt.plot(feature)


epochs = 50
bs = 512
features = 1
length = 200
order = 5
samples = 6*bs
hidden_features = 1
num_layers=5
lr = 5e-3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = nn.RNN(features, hidden_features, num_layers, batch_first=True).to(device)

ds = TSDataSet(features=features, order=order, length=length, samples=samples)
dl = DataLoader(ds, batch_size=bs, num_workers=8, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.MSELoss()

# Training
for epoch in range(epochs):
    if epoch+1 % 10 == 0:
        for g in optimizer.param_groups:
            g['lr'] /= 2
    for batch in dl:
        hidden = torch.zeros((num_layers, bs, hidden_features), device=device)
        batch = batch.to(device)
        loss = 0
        optimizer.zero_grad()

        data = batch[:, :-1, :]
        target = batch[:, -1, :]
        out, hn = model(data, hidden)
        loss = criterion(out[:, -1, :], target)
        loss.backward()
        optimizer.step()


# Testing the model
bs = 1
samples = bs
ds = TSDataSet(features=features, order=order, length=length, samples=samples)
dl = DataLoader(ds, batch_size=bs, num_workers=8, shuffle=False)

with torch.no_grad():
    predictions = torch.zeros((samples, length, features))
    truth = torch.zeros((samples, length, features))
    for batch in dl:
        hidden = torch.zeros((num_layers, bs, hidden_features), device=device)
        batch = batch.to(device)
        truth[:bs] = batch
        predictions[:bs] = batch

        data = batch[:, :-1, :]
        out, hn = model(data, hidden)
        predictions[:bs, -1, :] = out[:, -1, :]

fig, ax = plt.subplots()
plot_batch(truth)
plot_batch(predictions)
plt.show()

