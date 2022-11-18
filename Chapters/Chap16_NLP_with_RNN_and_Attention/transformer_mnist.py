import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import datasets, transforms

from transformers import PositionEmbedding, EncoderBlock

class Model(nn.Module):
    def __init__(self, max_length, dim, model_dim=64, stack=6, heads=8, dropout=0.1, d_ff=256, n_classes=10):
        super(Model, self).__init__()

        self.pos_embedding = PositionEmbedding(max_length, model_dim)
        #self.embedding = nn.Linear(dim, model_dim)
        self.embedding = nn.Conv1d(dim, model_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        self.encoder = nn.ModuleList(
                [
                    EncoderBlock(model_dim, heads, dropout, d_ff)
                    for _ in range(stack)
                ]
        )

        self.flatten = nn.Flatten(-2, -1)
        self.lin1 = nn.Linear(max_length * model_dim, model_dim)
        self.out = nn.Linear(model_dim, n_classes)

    def forward(self, x):
        # If run conv1d -> input need shape (N, C, L) and not (N, L, C) as it currently is -> transpose
        x = self.embedding(x.transpose(-2, -1)).transpose(-2, -1)
        #x = self.embedding(x)
        x = self.dropout(x + self.pos_embedding(x))
        for encoder_layer in self.encoder:
            x = encoder_layer(x)

        x = self.activation(self.lin1(self.flatten(x)))
        x = self.out(x)
        return x

def get_mnist(folder, download=False):
    transform = transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(dtype=torch.float32)])
    train_dataset = datasets.MNIST(root=folder, train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST(root=folder, train=False, download=download, transform=transform)
    return train_dataset, test_dataset


def train(model, train_dl, test_dl, max_epochs, learning_rate, device='cpu'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    for epoch in range(max_epochs):
        for idx, batch in enumerate(train_dl):
            data, target = batch[0].squeeze().to(device), batch[1].to(device)

            logits = model(data)

            loss = criterion(logits, target)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if idx % 20 == 0:
                acc = torch.sum(torch.argmax(torch.softmax(logits, 1), 1) == target)/logits.shape[0]
                print(f'epoch: {epoch}, batch: {idx}, loss: {loss:.2f}, acc: {acc*100:.1f}%')

        with torch.no_grad():
            val_loss = 0
            correct = 0
            items = 0
            for idx, batch in enumerate(test_dl):
                data, target = batch[0].squeeze().to(device), batch[1].to(device)

                # Forward pass
                logits = model(data)

                # Compute loss
                l = criterion(logits, target)
                val_loss += l
                correct += torch.sum(torch.argmax(torch.softmax(logits, 1), 1) == target)
                items += data.shape[0]

            print(f'valid_loss: {val_loss:.2f}, acc: {(correct/items)*100:.1f}%')

        scheduler.step(val_loss)
    return model


if __name__ == '__main__':
    data_folder = './data/'
    device = 'cuda'
    batch_size = 512
    num_workers = 8
    max_epochs = 20
    learning_rate = 0.01
    save_model_path = 'mnist_transformer.ckpt'

    # Transformer params
    model_dim = 64
    stack = 6
    dropout = 0.1
    heads = 8
    d_ff = 256

    train_ds, test_ds = get_mnist(data_folder, download=True)
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    image_shape = next(iter(test_dl))[0][0].shape
    max_length, dim = image_shape[-2], image_shape[-1]

    model = Model(max_length, dim, stack=stack, model_dim=model_dim, heads=heads, dropout=dropout, d_ff=d_ff)

    # Test sending data through model shape: [N,T,C]
    model(next(iter(test_dl))[0].squeeze())

    trained_model = train(model, train_dl, test_dl, max_epochs, learning_rate, device=device)
