import torch
import torch.nn as nn

class BaseAutoencoder(nn.Module):
    def __init__(self):
        super(BaseAutoencoder, self).__init__()
        pass

    def encode(self, x):
        raise NotImplementedError

    def decode(self, x):
        raise NotImplementedError

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded 