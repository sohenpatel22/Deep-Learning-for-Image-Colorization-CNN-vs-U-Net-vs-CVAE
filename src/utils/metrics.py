import torch
import torch.nn.functional as F


def mse_metric(y_pred, y_true):
    """
    Mean Squared Error
    """
    return F.mse_loss(y_pred, y_true).item()


def psnr(y_pred, y_true):
    """
    Peak Signal-to-Noise Ratio
    """
    mse = F.mse_loss(y_pred, y_true)
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


def ssim(y_pred, y_true):
    """
    Simplified SSIM (basic version)
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = y_pred.mean()
    mu_y = y_true.mean()

    sigma_x = y_pred.var()
    sigma_y = y_true.var()
    sigma_xy = ((y_pred - mu_x) * (y_true - mu_y)).mean()

    ssim_value = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    )

    return ssim_value.item()