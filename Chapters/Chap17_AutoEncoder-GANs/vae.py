import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def get_mnist(folder, download=False):
    transform = transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(dtype=torch.float32)])
    train_dataset = datasets.MNIST(root=folder, train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST(root=folder, train=False, download=download, transform=transform)
    return train_dataset, test_dataset


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dim=2):
        super(VariationalEncoder, self).__init__()

        self.linear1 = nn.Linear(28**2, 512)
        self.mu = nn.Linear(512, latent_dim)
        self.sigma = nn.Linear(512, latent_dim)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()

        self.kl = 0.0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))

        mu = self.mu(x)
        sigma = torch.exp(self.sigma(x))

        z = mu + sigma * self.N.sample(mu.shape)

        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()

        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim=2):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dim, 512)
        self.linear2 = nn.Linear(512, 28*28)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x.view(-1, 1, 28, 28)

class Model(nn.Module):
    def __init__(self, latent_dim=2):
        super(Model, self).__init__()
        self.encoder = VariationalEncoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)


def train(model: nn.Module, data: DataLoader, valid_data: DataLoader=None, lr=0.001, max_epochs=30, device='cpu'):
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(max_epochs):
        for i, (x, y) in enumerate(data):
            x = x.to(device)

            optimizer.zero_grad()

            logits = model(x)

            loss = ((x - logits)**2).sum() + model.encoder.kl

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
    return model


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
        im = ax.scatter(latent_var[:, 0], latent_var[:, 1], c=y, cmap='tab20')

    fig.colorbar(im)

    if filename: plt.savefig(filename)
    plt.close(fig)

@torch.no_grad()
def plot_reconstructed(model, r0=(-5, 10), r1=(-10, 5), n=12, filename=None, device='cpu'):
    model = model.to(device)
    w = 28
    img = np.zeros((n*w, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = model.decoder(z)
            x_hat = x_hat.reshape(28, 28).cpu().detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    im = ax.imshow(img, extent=[*r0, *r1])
    if filename: plt.savefig(filename)
    plt.close(fig)

@torch.no_grad()
def interpolate_gif(model, x_1, x_2, filename, n=100):
    z_1 = model.encoder(x_1)
    z_2 = model.encoder(x_2)

    z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n)])

    interpolate_list = model.decoder(z)
    interpolate_list = interpolate_list.cpu().detach().numpy()*255

    images_list = [Image.fromarray(img.reshape(28, 28)).resize((256, 256)) for img in interpolate_list]
    images_list = images_list + images_list[::-1] # loop back beginning

    images_list[0].save(
        f'{filename}.gif',
        save_all=True,
        append_images=images_list[1:],
        loop=1)


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
    model = train(model, train_dl, valid_data=test_dl, device=device)
    x, y = next(iter(test_dl))
    x, y = x.to(device), y.to(device)
    plot_data([x, model(x)], filename='./trained_predictions.png')
    plot_latent_space(model, train_dl, filename='./trained_latent_space.png', device=device)
    plot_reconstructed(model, filename='./reconstructed.png', device=device)

    x_1 = x[y == 1][0]
    x_2 = x[y == 0][0]
    interpolate_gif(model, x_1, x_2, './autoencoder_interp')
