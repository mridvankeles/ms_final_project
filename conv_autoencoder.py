import torch
import torch.nn as nn
from base_autoencoder import BaseAutoencoder

class ConvAutoencoder(BaseAutoencoder):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 256, 3, stride=2 , padding=1 ),  

            nn.LeakyReLU(0.2),            
            nn.Conv2d(256, 64, 3, stride=2 , padding=1 ),  

            nn.LeakyReLU(0.2),            
            nn.Conv2d(64, 32, 3, stride=2 , padding=1 ),  

            nn.LeakyReLU(0.2),            
            nn.Conv2d(32, 3, 3, stride=2 , padding=1 ),  #3x8x8
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(3, 32, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(32, 64, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 256, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(256, 3, 3, stride=2, padding=1, output_padding=1),
        )

    def encode(self, x):
        #compressed remask features
        return self.encoder(x)
    

    def decode(self, x):
        return self.decoder(x)

    def load_data(self, features_path):
        features = torch.load(features_path)
        return features