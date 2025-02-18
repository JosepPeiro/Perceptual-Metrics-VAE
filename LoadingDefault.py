import os
from preprocessing import Preprocessing, SplitAudio
import torchaudio

import torch
from torch.utils.data import DataLoader, TensorDataset

from torch import nn
import torch.optim as optim

from torchinfo import summary


def LoadAudios(route = "./MusicCaps", limit = None):

    archivos = os.listdir(route)

    X = []
    metadata = []
    sample_rate_red = 16000

    if limit is not None:
        archivos = archivos[:limit]

    for archivo in archivos:
        waveform, samp_rt = torchaudio.load(route + "/" + archivo)
        f, s = SplitAudio(waveform, sample_rate = samp_rt, new_sample_rate = sample_rate_red)

        f_spec, f_maxi, f_mini = Preprocessing(f, 16000, resampler_f = False)
        s_spec, s_maxi, s_mini = Preprocessing(s, 16000, resampler_f = False)

        X += [f_spec, s_spec]
        metadata += [{"nombre":archivo, "parte":"first", "minimum":f_mini, "maximum":f_maxi},
                    {"nombre":archivo, "parte":"second", "minimum":s_mini, "maximum":s_maxi}]
        
    return X, metadata


def Tensoring(X, batch_size = 64):
    X_tensor = torch.stack(X)  # Suponiendo que X es una lista de tensores (N, 256, 256)
    # Añadir la dimensión del canal (1 para monocanal)
    X_tensor = X_tensor.unsqueeze(1)  # (N, 1, 256, 256)

    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def LoadData(route = "./MusicCaps", limit = None, batch_size = 64):
    X, _ = LoadAudios(route = route, limit=limit)
    return Tensoring(X, batch_size = batch_size)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=2, padding=1),  # Reduce tamaño a 128x128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),  # Reduce tamaño a 128x128
            nn.BatchNorm2d(128),
            nn.Tanh()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded