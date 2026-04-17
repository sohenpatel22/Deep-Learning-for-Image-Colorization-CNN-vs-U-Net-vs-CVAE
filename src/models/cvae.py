import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim=64, num_filters=32, kernel_size=3):
        super().__init__()

        padding = kernel_size // 2
        in_channels = 4   # 1 grayscale + 3 RGB

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),

            nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 32 -> 16

            nn.Conv2d(num_filters, num_filters * 2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(),

            nn.Conv2d(num_filters * 2, num_filters * 2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(),
            nn.MaxPool2d(2)    # 16 -> 8
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_filters * 2 * 8 * 8, 256),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x_grey, y_rgb):
        # concatenate along channel dimension
        x = torch.cat([x_grey, y_rgb], dim=1)   # shape: (B, 4, 32, 32)
        x = self.conv(x)
        x = self.fc(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=64, num_filters=32, kernel_size=3):
        super().__init__()

        padding = kernel_size // 2

        # encode grayscale image to feature map
        self.x_encoder = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 32 -> 16

            nn.Conv2d(num_filters, num_filters * 2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(),
            nn.MaxPool2d(2)    # 16 -> 8
        )

        # convert latent vector into feature map
        self.z_to_feat = nn.Sequential(
            nn.Linear(latent_dim, num_filters * 2 * 8 * 8),
            nn.ReLU()
        )

        # decode back to RGB image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_filters * 4, num_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(),

            nn.ConvTranspose2d(num_filters * 2, num_filters, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),

            nn.Conv2d(num_filters, 3, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid()
        )

    def forward(self, x_grey, z):
        x_feat = self.x_encoder(x_grey)   # (B, 2F, 8, 8)
        z_feat = self.z_to_feat(z)        # (B, 2F*8*8)
        z_feat = z_feat.view(z.size(0), -1, 8, 8)   # (B, 2F, 8, 8)

        h = torch.cat([x_feat, z_feat], dim=1)      # (B, 4F, 8, 8)
        y_hat = self.decoder(h)                     # (B, 3, 32, 32)

        return y_hat


class CVAE(nn.Module):
    def __init__(self, latent_dim=64, num_filters=32, kernel_size=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim=latent_dim, num_filters=num_filters, kernel_size=kernel_size)
        self.decoder = Decoder(latent_dim=latent_dim, num_filters=num_filters, kernel_size=kernel_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x_grey, y_rgb):
        mu, logvar = self.encoder(x_grey, y_rgb)
        z = self.reparameterize(mu, logvar)
        y_hat = self.decoder(x_grey, z)

        return y_hat, mu, logvar

    @torch.no_grad()
    def sample(self, x_grey, n_samples=1, logvar=0.0):
        self.eval()
        batch_size = x_grey.size(0)
        device = x_grey.device

        std = torch.exp(torch.tensor(0.5 * logvar, device=device))
        z = torch.randn(batch_size * n_samples, self.latent_dim, device=device) * std

        x_repeat = x_grey.repeat_interleave(n_samples, dim=0)
        y_samples = self.decoder(x_repeat, z)
        y_samples = y_samples.view(batch_size, n_samples, 3, 32, 32)

        return y_samples


def cvae_loss(y_hat, y_true, mu, logvar, beta=1.0):
    # reconstruction loss
    recon_loss = F.mse_loss(y_hat, y_true, reduction="mean")

    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


if __name__ == "__main__":
    batch_size = 8
    latent_dim = 64

    x_grey = torch.randn(batch_size, 1, 32, 32)
    y_rgb = torch.randn(batch_size, 3, 32, 32)

    model = CVAE(latent_dim=latent_dim, num_filters=32, kernel_size=3)

    y_hat, mu, logvar = model(x_grey, y_rgb)
    loss, recon, kl = cvae_loss(y_hat, y_rgb, mu, logvar)

    print("x_grey shape:", x_grey.shape)
    print("y_rgb shape:", y_rgb.shape)
    print("y_hat shape:", y_hat.shape)
    print("mu shape:", mu.shape)
    print("logvar shape:", logvar.shape)
    print("total loss:", loss.item())
    print("recon loss:", recon.item())
    print("kl loss:", kl.item())