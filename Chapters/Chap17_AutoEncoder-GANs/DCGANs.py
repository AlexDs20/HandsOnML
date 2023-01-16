import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as TF
from torchvision.datasets import FashionMNIST

import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self, embed_dims: int=30):
        super(Generator, self).__init__()
        """
        [N, embed_dims] -> [N, 98, 1, 1] -> [N, 2, 7, 7] -> [N, 4, 7, 7] -> [N, 16, 7, 7] -> [N, 32, 14, 14] -> [N, 64, 28, 28] -> [N, 1, 28, 28]
        """
        self.layers = nn.Sequential(
            nn.Linear(embed_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 98),
            nn.ReLU(),
            nn.Unflatten(-1, (2, 7, 7)),
            nn.Conv2d(2, 4, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding='same'),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 4, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Flatten(), # -> 4 * 7 * 7
            nn.Linear(4*7*7, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


def train_gan(generator, discriminator, train_dl, max_epochs=30, device='cpu', lr=0.001, embed_dims=30, *args, **kwargs):
    criterion = nn.BCELoss()
    G_optim = torch.optim.RMSprop(generator.parameters(), lr=lr)
    D_optim = torch.optim.RMSprop(discriminator.parameters(), lr=lr)


    generator = generator.to(device)
    discriminator = discriminator.to(device)

    for epoch in range(max_epochs):
        for i, (x, _) in enumerate(train_dl):
            x = x.to(device)
            # Train Discriminator
            '''
            - Create noise
            - Run through generator
            - Combine fake and real data
            - Run prediction through Discriminator
            - zero_grad()
            - Compute loss
            - Update weights
            '''
            noise = torch.randn(x.shape[0], embed_dims, device=device)
            fake = generator(noise)
            stacked = torch.concat([x, fake], dim=0)
            labels = torch.concat([torch.ones(x.shape[0], 1, device=device),
                                   torch.zeros(fake.shape[0], 1, device=device)], dim=0)
            logits = discriminator(stacked)

            loss = criterion(logits, labels)

            G_optim.zero_grad()
            D_optim.zero_grad()
            loss.backward()
            D_optim.step()
            print('--------------------')
            print(loss.item())

            # Train Generator
            '''
            - Create noise
            - Send through Generatorn
            - Send through Discriminator
            - Compute discriminator and give the label to be a true image
            - Compute loss
            - update generator weights only
            '''
            noise = torch.randn(x.shape[0], embed_dims, device=device)
            fake = generator(noise)
            labels = torch.ones(fake.shape[0], 1, device=device)

            logits = discriminator(fake)
            loss = criterion(logits, labels)

            G_optim.zero_grad()
            D_optim.zero_grad()
            loss.backward()
            G_optim.step()

            print(loss.item())

    return generator, discriminator

def load_data(root='./data/', download=False):
    transforms = TF.ToTensor()
    train_ds = FashionMNIST(root=root, download=download, transform=transforms, train=True)
    test_ds = FashionMNIST(root=root, download=download, transform=transforms, train=False)
    return train_ds, test_ds


def main():
    batch_size = 512
    num_workers = 8
    max_epochs = 100
    embed_dims = 30
    device = 'cpu'
    max_epochs = 30

    train_ds, test_ds = load_data()
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    generator = Generator(embed_dims=embed_dims)
    discriminator = Discriminator()
    noise = torch.randn((256, 30))
    out = generator(noise)
    out = discriminator(out)
    out = discriminator(next(iter(test_dl))[0])

    train_gan(generator, discriminator, train_dl, max_epochs=max_epochs, device=device)

if __name__ == '__main__':
    main()
