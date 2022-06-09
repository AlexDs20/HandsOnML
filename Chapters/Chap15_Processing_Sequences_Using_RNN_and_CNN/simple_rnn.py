import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplestRNN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(SimplestRNN, self).__init__()
        self.hidden = self._init_hidden(hidden_features)

        self.i2h = nn.Linear(in_features, hidden_features)
        self.h2h = nn.Linear(hidden_features, hidden_features)


    def forward(self, x):
        out = F.gelu(self.i2h(x) + self.h2h(self.hidden))
        self.hidden = out
        return out

    def _init_hidden(self, hidden_features):
        return torch.zeros((hidden_features))



if __name__ == '__main__':
    bs = 5
    features = 13
    hidden_features = 10

    x = torch.ones((features))
    model = SimplestRNN(features, hidden_features)
    out = model(x)
    print(out)
