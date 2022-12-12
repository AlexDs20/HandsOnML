import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import datasets, transforms


class Noise(object):
    def __init__(self, mean, std=1.):
        super(Noise, self).__init__()
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(x.size()) * self.std + self.mean
        return x + noise.to(x.device)

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.layers = nn.ModuleList([
            nn.Conv2d(1, 16, kernel_size=3, padding='same'),        # -> 16x28x28
            nn.Conv2d(16, 16, kernel_size=3, padding='same'),        # -> 16x28x28
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),                              # -> 16x14x14
            nn.Conv2d(16, 32, kernel_size=3, padding='same'),       # -> 32x14x14
            nn.Conv2d(32, 32, kernel_size=3, padding='same'),       # -> 32x14x14
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),                              # -> 32x7x7
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),       # -> 64x7x7
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),       # -> 64x7x7
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),    # -> 32x14x14
            nn.Conv2d(32, 16, kernel_size=3, padding='same'),       # -> 16x14x14
            nn.Conv2d(16, 16, kernel_size=3, padding='same'),       # -> 16x14x14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),     # -> 8x28x28
            nn.Conv2d(8, 8, kernel_size=3, padding='same'),         # -> 1x28x28
            nn.Conv2d(8, 1, kernel_size=3, padding='same'),         # -> 1x28x28
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def get_mnist(folder, download=False):
    transform = transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(dtype=torch.float32)])
    train_dataset = datasets.MNIST(root=folder, train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST(root=folder, train=False, download=download, transform=transform)
    return train_dataset, test_dataset


def train(model, train_dl, test_dl, max_epochs, learning_rate, device='cpu', transform=lambda x: x):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1)

    for epoch in range(max_epochs):
        for idx, batch in enumerate(train_dl):
            data, target = batch[0].to(device), batch[1].to(device)

            if torch.randint(2, (1,)).item() == 1:
                mean = torch.rand(1)
                std = torch.rand(1)
                transform = Noise(mean, std)
                transform_data = transform(data)
            else:
                transform_data = data

            logits = model(transform_data)

            loss = criterion(logits, data)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if idx % 20 == 0:
                print(f'epoch: {epoch}, batch: {idx}, loss: {loss:.2f}')

        with torch.no_grad():
            val_loss = 0

            for idx, batch in enumerate(test_dl):
                data, target = batch[0].to(device), batch[1].to(device)
                if torch.randint(2, (1,)).item() == 1:
                    mean = torch.rand(1)
                    std = torch.rand(1)
                    transform = Noise(mean, std)
                    transform_data = transform(data)
                else:
                    transform_data = data

                # Forward pass
                logits = model(transform_data)

                # Compute loss
                l = criterion(logits, data)
                val_loss += l

            plot_data(model, data[:8], transform, epoch=epoch)
            print(f'valid_loss: {val_loss:.2f}')

        scheduler.step(val_loss)
    return model


def plot_data(model, data, transform, epoch=0):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(data.shape[0], 3)

    transform_data= transform(data)
    model.to(data.device)
    predictions = model(transform_data)

    for i in range(data.shape[0]):
        ax[i, 0].imshow(data[i, 0].squeeze().cpu().detach().numpy())
        ax[i, 1].imshow(transform_data[i, 0].squeeze().cpu().detach().numpy())
        ax[i, 2].imshow(predictions[i, 0].squeeze().cpu().detach().numpy())

    plt.savefig(f'/home/alexandre/Desktop/predictions_{epoch}.png')


if __name__ == '__main__':
    data_folder = '../Chap16_NLP_with_RNN_and_Attention/data/'
    device = 'cuda'
    batch_size = 256
    num_workers = 8
    max_epochs = 15
    learning_rate = 0.001
    save_model_path = 'mnist_num_workers.ckpt'
    transform = Noise(0.8, 0.8)

    train_ds, test_ds = get_mnist(data_folder, download=False)
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    model = Model()

    # Test sending data through model
    model(next(iter(test_dl))[0])

    trained_model = train(model, train_dl, test_dl, max_epochs, learning_rate, device=device, transform=transform)

    #data, _ = next(iter(test_dl))
    #plot_data(model, data[:8], transform)
