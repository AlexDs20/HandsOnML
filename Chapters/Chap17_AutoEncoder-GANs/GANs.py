import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as TF
from torchvision.datasets import FashionMNIST

import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self, embed_dims=30, out_shape=(28,28)):
        super(Generator, self).__init__()
        self.out_shape = out_shape
        self.layers = nn.ModuleList(
            [
                nn.Linear(embed_dims, 100),
                nn.SELU(),
                nn.Linear(100, 150),
                nn.SELU(),
                nn.Linear(150, out_shape[0]*out_shape[1]),
                nn.Sigmoid(),
            ]
        )


    def forward(self, x):
        """
        Take in random values and produces an image
        """
        shape = x.shape
        for l in self.layers:
            x = l(x)
        x = x.reshape(shape[0], 1, *self.out_shape)
        return x

class Discriminator(nn.Module):
    def __init__(self, shape=(28,28)):
        super(Discriminator, self).__init__()
        self.layers = nn.ModuleList(
            [
                nn.Flatten(start_dim=1),
                nn.Linear(shape[0]*shape[1], 150),
                nn.SELU(),
                nn.Linear(150, 100),
                nn.SELU(),
                nn.Linear(100, 1),
                nn.Sigmoid(),
            ]
        )

    def forward(self, x):
        for l in self.layers:
            x = l(x)

        return x


def train_gan(generator, discriminator, train_dl, test_dl, max_epochs=30, lr=0.001, embed_dims=30, device='cpu'):
    criterion = torch.nn.BCELoss()
    D_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=lr)
    G_optimizer = torch.optim.RMSprop(generator.parameters(), lr=lr)

    for epoch in range(max_epochs):
        dl = 0
        gl = 0
        for i, (x, y) in enumerate(train_dl):
            # Train the Discriminator
            x, y = x.to(device), y.to(device)
            noise = torch.randn((x.shape[0], embed_dims), device=device)

            fake = generator(noise[::2])
            stacked = torch.concat([x[::2], fake], dim=0)

            pred = discriminator(stacked)

            loss = criterion(pred, torch.concat([torch.ones(x[::2].shape[0], 1, device=device), torch.zeros(fake.shape[0], 1, device=device)], dim=0) )

            G_optimizer.zero_grad()
            D_optimizer.zero_grad()
            loss.backward()
            D_optimizer.step()

            dl += loss

            # Train the Generator
            noise = torch.randn((x.shape[0], embed_dims), device=device)
            fake = generator(noise)
            pred = discriminator(fake)
            loss = criterion(pred, torch.ones(x.shape[0], 1, device=device))

            D_optimizer.zero_grad()
            G_optimizer.zero_grad()
            loss.backward()
            G_optimizer.step()

            gl += loss

        print(f'{epoch}: {dl.item()/(i+1):.4f}, {gl.item()/(i+1):.4f}')

        save_image(fake, epoch)

    return generator, discriminator


def save_image(data, epoch, N=8):
    fig, ax = plt.subplots(N, 1)

    for i in range(N):
        ax[i].imshow(data[i,0].cpu().detach().numpy(), cmap='gray', vmin=0, vmax=1)

    plt.savefig(f'generated/{epoch}.png')
    plt.close()



def get_dataset(folder='./data/', download=False):
    transform = TF.ToTensor()
    train_data = FashionMNIST(root=folder, train=True, transform=transform, download=download)
    test_data = FashionMNIST(root=folder, train=False, transform=transform, download=download)

    return train_data, test_data


def main():
    batch_size = 256
    num_workers = 8
    max_epochs = 1000
    embed_dims = 40
    device = 'cuda'

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    train_ds, test_ds = get_dataset(download=False)

    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    gen_model = Generator(embed_dims).to(device)
    disc_model = Discriminator().to(device)

    train_gan(gen_model, disc_model, train_dl, test_dl, max_epochs=max_epochs, embed_dims=embed_dims, device=device)

if __name__ == '__main__':
    main()
