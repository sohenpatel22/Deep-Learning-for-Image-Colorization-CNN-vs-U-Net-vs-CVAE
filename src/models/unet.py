import torch
import torch.nn as nn


class UNet(nn.Module):
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

        # Decoder with skip connections
        self.upconv1 = nn.Sequential(
            nn.Conv2d(num_filters * 4, num_filters, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )

        self.upconv2 = nn.Sequential(
            nn.Conv2d(num_filters * 2, 3, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )

        # final layer also takes original grayscale input
        self.finalconv = nn.Conv2d(4, 3, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        # Encoder
        down1 = self.downconv1(x)     # (B, F, 16, 16)
        down2 = self.downconv2(down1) # (B, 2F, 8, 8)

        # Bottleneck
        rf = self.rfconv(down2)       # (B, 2F, 8, 8)

        # Skip connection 1: concatenate rf and down2
        up1_input = torch.cat([rf, down2], dim=1)   # (B, 4F, 8, 8)
        up1 = self.upconv1(up1_input)               # (B, F, 16, 16)

        # Skip connection 2: concatenate up1 and down1
        up2_input = torch.cat([up1, down1], dim=1)  # (B, 2F, 16, 16)
        up2 = self.upconv2(up2_input)               # (B, 3, 32, 32)

        # Final skip connection: concatenate decoder output and original input
        final_input = torch.cat([up2, x], dim=1)    # (B, 4, 32, 32)
        out = self.finalconv(final_input)           # (B, 3, 32, 32)

        return out


if __name__ == "__main__":
    model = UNet(kernel_size=3, num_filters=32)
    x = torch.randn(8, 1, 32, 32)
    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)