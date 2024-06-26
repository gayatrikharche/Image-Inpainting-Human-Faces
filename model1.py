import torch
import torch.nn as nn

class Autoencoder1(nn.Module):
    def __init__(self):
        super(Autoencoder1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), # 124x124 -> 124x124
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2), # 124x124 -> 62x62
            nn.Conv2d(16, 8, kernel_size=3, padding=1), # 62x62 -> 62x62
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2), # 62x62 -> 31x31
            nn.Conv2d(8, 8, kernel_size=3, padding=1), # 31x31 -> 31x31
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2) # 31x31 -> 16x16
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1), # 16x16 -> 16x16
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'), # 16x16 -> 32x32
            nn.Conv2d(8, 8, kernel_size=3, padding=1), # 32x32 -> 32x32
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'), # 32x32 -> 64x64
            nn.Conv2d(8, 16, kernel_size=3, padding=1), # 64x64 -> 64x64
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'), # 64x64 -> 128x128
            nn.Conv2d(16, 3, kernel_size=3, padding=1), # 128x128 -> 128x128
            nn.Sigmoid(),
            nn.Upsample(size=(124, 124), mode='bilinear', align_corners=False) # Ensure final size is 124x124
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x