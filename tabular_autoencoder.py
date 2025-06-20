import torch
import torch.nn as nn
import pandas as pd
from base_autoencoder import BaseAutoencoder
from sklearn.preprocessing import StandardScaler


class TabularAutoencoder(BaseAutoencoder):
    def __init__(self, input_dim=19, encoding_dim=3):
        super(TabularAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, encoding_dim),

        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def load_data(self, metric_df):
        scaler = StandardScaler()
        tabular_scaled = scaler.fit_transform(metric_df)  # df shape (2882, 19)
        # Example: Select relevant columns and convert to tensor
        data = torch.tensor(tabular_scaled).float()
        return data