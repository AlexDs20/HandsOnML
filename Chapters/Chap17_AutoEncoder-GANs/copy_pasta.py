import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 784)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))

class Autoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        #sigma = torch.exp(self.linear3(x))
        log_var = torch.exp(self.linear3(x))
        # z = mu + sigma * self.N.sample(mu.shape)
        z = mu + torch.exp(log_var/2) * self.N.sample(mu.shape)
        # self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        self.kl = -0.5 * (1 + log_var - torch.exp(log_var) - mu**2).sum(dim=1).mean(dim=0)
        return z

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def train(autoencoder, data, epochs=15):
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        print(epoch)
        for x, y in data:
            x = x.to(device) # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum()
            loss.backward()
            opt.step()
        plot_data(autoencoder, x[:8], epoch=epoch)
    return autoencoder

def plot_latent(autoencoder, data, num_batches=50):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        fig, ax = plt.subplots(1, 1, figsize=(15,15))
        im = ax.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax)
            break
    fig.savefig('/mnt/d/tmp/image.png')
    plt.close(fig)

def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
    w = 28
    img = np.zeros((n*w, n*w))
    fig, ax = plt.subplots(1, 1, figsize=(15,15))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    ax.imshow(img, extent=[*r0, *r1])
    plt.savefig('/mnt/d/tmp/reconstructed.png')
    plt.close()

def plot_data(model, batch, epoch=0):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(batch.shape[0], 2, figsize=(20,20))

    model.to(batch.device)
    predictions = model(batch)

    for i in range(batch.shape[0]):
        ax[i, 0].imshow(batch[i, 0].squeeze().cpu().detach().numpy())
        ax[i, 1].imshow(predictions[i, 0].squeeze().cpu().detach().numpy())

    plt.savefig(f'/mnt/d/tmp/predictions_{epoch}.png')

def interpolate(autoencoder, x_1, x_2, n=12):
    z_1 = autoencoder.encoder(x_1)
    z_2 = autoencoder.encoder(x_2)
    z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n)])
    interpolate_list = autoencoder.decoder(z)
    interpolate_list = interpolate_list.to('cpu').detach().numpy()

    w = 28
    img = np.zeros((w, n*w))
    for i, x_hat in enumerate(interpolate_list):
        img[:, i*w:(i+1)*w] = x_hat.reshape(28, 28)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])

def interpolate_gif(autoencoder, filename, x_1, x_2, n=100):
    z_1 = autoencoder.encoder(x_1)
    z_2 = autoencoder.encoder(x_2)

    z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n)])

    interpolate_list = autoencoder.decoder(z)
    interpolate_list = interpolate_list.to('cpu').detach().numpy()*255

    images_list = [Image.fromarray(img.reshape(28, 28)).resize((256, 256)) for img in interpolate_list]
    images_list = images_list + images_list[::-1] # loop back beginning

    images_list[0].save(
        f'{filename}.gif',
        save_all=True,
        append_images=images_list[1:],
        loop=1)

if __name__ == "__main__":
    latent_dims = 2
    #autoencoder = Autoencoder(latent_dims).to(device) # GPU
    autoencoder = VariationalAutoencoder(latent_dims).to(device)

    data = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('../Chap16_NLP_with_RNN_and_Attention/data/',
                   transform=torchvision.transforms.ToTensor(),
                   download=True),
            batch_size=512,
            shuffle=True)

    autoencoder = train(autoencoder, data)

    plot_latent(autoencoder, data)
    plot_reconstructed(autoencoder, r0=[-20, 20], r1=[-20, 20])

    x, y = next(iter(data))
    x_1 = x[y == 1][1].to(device) # find a 1
    x_2 = x[y == 0][1].to(device) # find a 0

    interpolate(autoencoder, x_1, x_2, n=20)
    interpolate_gif(autoencoder, "/mnt/d/tmp/autoencoder", x_1, x_2)
