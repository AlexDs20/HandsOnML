import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as TF
from torchvision.datasets import FashionMNIST


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
                nn.Linear(100, 1)
            ]
        )

    def forward(self, x):
        for l in self.layers:
            x = l(x)

        return x


def get_dataset(folder='./data/', download=False):
    transform = TF.ToTensor()
    train_data = FashionMNIST(root=folder, train=True, transform=transform, download=download)
    test_data = FashionMNIST(root=folder, train=False, transform=transform, download=download)

    return train_data, test_data


def main():
    batch_size = 256
    num_workers = 8

    train_ds, test_ds = get_dataset(download=False)

    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    gen_model = Generator()
    disc_model = Discriminator()

    #train_gan(gen_model, disc_model, train_dl, test_dl)

if __name__ == '__main__':
    main()
