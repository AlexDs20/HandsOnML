import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt


def get_mnist(folder, download=False):
    transform = transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(dtype=torch.float32)])
    train_dataset = datasets.MNIST(root=folder, train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST(root=folder, train=False, download=download, transform=transform)
    return train_dataset, test_dataset


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dim=2):
        super(VariationalEncoder, self).__init__()

        self.linear1 = nn.Linear(28**2, 512)
        self.linear2 = nn.Linear(512, 512)
        self.mu = nn.Linear(512, latent_dim)
        self.sigma = nn.Linear(512, latent_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))
        sigma = self.sigma(x)
        mu = self.mu(x)

        z = mu + sigma * torch.randn(*sigma.shape, device=sigma.device)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim=2):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(2, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 28*28)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))
        x = self.act(self.linear3(x))
        return x.view(-1, 1, 28, 28)

class Model(nn.Module):
    def __init__(self, latent_dim=2):
        super(Model, self).__init__()
        self.encoder = VariationalEncoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)


def train(model: nn.Module, data: DataLoader, valid_data: DataLoader=None, lr=0.001, max_epochs=50, device='cpu'):
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(max_epochs):
        for i, (x, y) in enumerate(data):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            logits = model(x)

            loss = torch.sqrt(((x - logits)**2).sum())

            loss.backward()

            optimizer.step()

        with torch.no_grad():
            if valid_data is not None:
                valid_loss = 0
                for i, (x, y) in enumerate(valid_data):
                    x, y = x.to(device), y.to(device)

                    logits = model(x)

                    valid_loss += torch.sqrt(((x - logits)**2).sum())

        print(f'epoch: {epoch}, loss: {loss:.3f}, valid loss: {valid_loss/len(valid_data):.3f}')


@torch.no_grad()
def plot_data(list_data, elements=8, filename=None):
    fig, ax = plt.subplots(elements, len(list_data))

    for i, ds in enumerate(list_data):
        for j, data in enumerate(ds):
            if j == elements: break
            ax[j, i].imshow(data[0].cpu().numpy())

    if filename: plt.savefig(filename)

@torch.no_grad()
def plot_latent_space(model, data, num_batches=2, device='cpu', filename=None):
    fig, ax = plt.subplots(1, 1)

    model.eval()
    model.to(device)
    for i, (x, y) in enumerate(data):
        if i == num_batches: break
        x = x.to(device)
        latent_var = model.encoder(x).cpu().numpy()
        im = ax.scatter(latent_var[:, 0], latent_var[:, 1], c=y, cmap='tab10')

    fig.colorbar(im)

    if filename: plt.savefig(filename)
    plt.close(fig)


if __name__ == '__main__':
    data_folder = '../Chap16_NLP_with_RNN_and_Attention/data/'
    device = 'cuda'
    batch_size = 256
    num_workers = 8

    train_ds, test_ds = get_mnist(data_folder, download=False)
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    model = Model().to(device)
    train(model, train_dl, valid_data=test_dl, device=device)
    d = next(iter(test_dl))[0].to(device)
    plot_data([d, model(d)], filename='./trained_predictions.png')
    plot_latent_space(model, train_dl, filename='./trained_latent_space.png', device=device)
