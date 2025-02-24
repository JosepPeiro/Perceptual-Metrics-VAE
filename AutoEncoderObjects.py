from LoadingDefault import LoadData

from torch import nn
import torch
import torch.nn.functional as F

from NLP.nlp import LaplacianPyramid

from pytorch_msssim import ms_ssim



class EntropyLimitedAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.centers = torch.Tensor([-1, 1])
        self.sigma = 10
        
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


    def encode(self, x):
        y = self.encoder(x)
        return y

    
    def quantise(self, y):
        if self.centers is None:
            return y
        y_flat = y.reshape(y.size(0), y.size(1), y.size(2)*y.size(3), 1)
        dist = torch.abs((y_flat - self.centers))**2
        if self.train:
            phi = F.softmax(-self.sigma * dist, dim=-1)
        else:
            phi = F.softmax(-1e7 * dist, dim=-1)
            symbols_hard = torch.argmax(phi, axis=-1)
            phi = F.one_hot(symbols_hard, num_classes=self.centers.size(0))
        inner_product = phi * self.centers
        y_hat = torch.sum(inner_product, axis=-1)
        y_hat = y_hat.reshape(y.shape)
        return y_hat
    

    def decode(self, y):
        x = self.decoder(y)
        return x
    

    def forward(self, x):
        encoded = self.encode(x)
        limit_entropy = self.quantise(encoded)
        decoded = self.decode(limit_entropy)
        return decoded



def ms_ssim_loss(x_pred, x_true):
    return 1 - ms_ssim(x_pred, x_true, data_range=1.0, size_average=True)

class MSSSIMLoss(nn.Module):
    def __init__(self):
        super(MSSSIMLoss, self).__init__()

    def forward(self, reconstructed, original):
        return 1 - ms_ssim(reconstructed, original, data_range=1.0, size_average=True)
    

class NLPDLoss(nn.Module):
    def __init__(self):
        super(NLPDLoss, self).__init__()
        self.lp = LaplacianPyramid(5, dims=1)
    def forward(self, reconstructed, original):
        return self.lp.compare(reconstructed, original)