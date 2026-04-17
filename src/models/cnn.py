import torch
import torch.nn as nn


class RegressionCNN(nn.Module):
    def __init__(self, kernel_size=3, num_filters=32):
        super().__init__()

        padding = kernel_size // 2

        # Encoder
        self.downconv1 = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.downconv2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Bottleneck
        self.rfconv = nn.Sequential(
            nn.Conv2d(num_filters * 2, num_filters * 2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU()
        )

        # Decoder
        self.upconv1 = nn.Sequential(
            nn.Conv2d(num_filters * 2, num_filters, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )

        self.upconv2 = nn.Sequential(
            nn.Conv2d(num_filters, 3, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )

        self.finalconv = nn.Conv2d(3, 3, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        x = self.downconv1(x)
        x = self.downconv2(x)
        x = self.rfconv(x)
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.finalconv(x)
        return x


if __name__ == "__main__":
    # quick test
    model = RegressionCNN(kernel_size=3, num_filters=32)
    x = torch.randn(8, 1, 32, 32)   # batch of 8 grayscale images
    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)