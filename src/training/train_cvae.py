import os
import sys
import time
import json
import numpy as np
import torch

# Add project root to path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data.loader import load_cifar10, get_batch
from src.data.preprocess import prepare_colourization_data
from src.models.cvae import CVAE, cvae_loss
from src.utils.visualization import show_images, plot_cvae_losses
from src.utils.metrics import mse_metric, psnr, ssim


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


def evaluate_cvae(model, test_grey, test_rgb, batch_size, device):
    """
    Evaluate CVAE reconstruction quality on test set.
    """
    model.eval()

    mse_scores = []
    psnr_scores = []
    ssim_scores = []

    with torch.no_grad():
        for batch_x, batch_y in get_batch(test_grey, test_rgb, batch_size):
            batch_x = torch.tensor(batch_x, dtype=torch.float32).to(device)
            batch_y = torch.tensor(batch_y, dtype=torch.float32).to(device)

            y_hat, _, _ = model(batch_x, batch_y)

            mse_scores.append(mse_metric(y_hat, batch_y))
            psnr_scores.append(psnr(y_hat, batch_y))
            ssim_scores.append(ssim(y_hat, batch_y))

    return np.mean(mse_scores), np.mean(psnr_scores), np.mean(ssim_scores)


def train_cvae(
    epochs=20,
    batch_size=50,
    learning_rate=0.001,
    num_filters=32,
    kernel_size=3,
    latent_dim=64,
    beta=0.5,
    device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device:", device)

    print("Loading CIFAR-10...")
    (x_train, y_train), (x_test, y_test) = load_cifar10()

    print("Preprocessing horse images...")
    train_rgb, train_grey = prepare_colourization_data(x_train, y_train)
    test_rgb, test_grey = prepare_colourization_data(x_test, y_test)

    print("Train RGB shape:", train_rgb.shape)
    print("Train Grey shape:", train_grey.shape)
    print("Test RGB shape:", test_rgb.shape)
    print("Test Grey shape:", test_grey.shape)

    model = CVAE(
        latent_dim=latent_dim,
        num_filters=num_filters,
        kernel_size=kernel_size
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_losses = []
    recon_losses = []
    kl_losses = []

    test_mse_scores = []
    test_psnr_scores = []
    test_ssim_scores = []

    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()

        epoch_total = []
        epoch_recon = []
        epoch_kl = []

        for batch_x, batch_y in get_batch(train_grey, train_rgb, batch_size):
            batch_x = torch.tensor(batch_x, dtype=torch.float32).to(device)
            batch_y = torch.tensor(batch_y, dtype=torch.float32).to(device)

            optimizer.zero_grad()

            y_hat, mu, logvar = model(batch_x, batch_y)
            loss, recon, kl = cvae_loss(y_hat, batch_y, mu, logvar, beta=beta)

            loss.backward()
            optimizer.step()

            epoch_total.append(loss.item())
            epoch_recon.append(recon.item())
            epoch_kl.append(kl.item())

        avg_total = np.mean(epoch_total)
        avg_recon = np.mean(epoch_recon)
        avg_kl = np.mean(epoch_kl)

        total_losses.append(avg_total)
        recon_losses.append(avg_recon)
        kl_losses.append(avg_kl)

        test_mse, test_psnr, test_ssim = evaluate_cvae(
            model, test_grey, test_rgb, batch_size, device
        )
        test_mse_scores.append(test_mse)
        test_psnr_scores.append(test_psnr)
        test_ssim_scores.append(test_ssim)

        print(
            f"Epoch [{epoch+1}/{epochs}] - "
            f"Total: {avg_total:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f} | "
            f"Test MSE: {test_mse:.4f} | Test PSNR: {test_psnr:.4f} | Test SSIM: {test_ssim:.4f}"
        )

        if avg_total < best_loss:
            best_loss = avg_total
            model_path = os.path.join(BASE_DIR, "outputs", "models", "cvae_best.pth")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to: {model_path}")

        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                sample_x = torch.tensor(test_grey[:5], dtype=torch.float32).to(device)
                sample_y = torch.tensor(test_rgb[:5], dtype=torch.float32).to(device)

                sample_pred, _, _ = model(sample_x, sample_y)

                show_images(
                    sample_x,
                    sample_y,
                    sample_pred,
                    num_images=5,
                    save_path=f"outputs/images/cvae_epoch_{epoch+1}.png"
                )

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")

    plot_cvae_losses(
        total_losses,
        recon_losses,
        kl_losses,
        save_path="outputs/plots/cvae_loss_curve.png"
    )

    print("\nFinal Metrics:")
    print(f"Best Total Loss: {min(total_losses):.4f}")
    print(f"Best Recon Loss: {min(recon_losses):.4f}")
    print(f"Best KL Loss: {min(kl_losses):.4f}")
    print(f"Best Test MSE: {min(test_mse_scores):.4f}")
    print(f"Best Test PSNR: {max(test_psnr_scores):.4f}")
    print(f"Best Test SSIM: {max(test_ssim_scores):.4f}")

    metrics = {
        "best_total_loss": float(min(total_losses)),
        "best_recon_loss": float(min(recon_losses)),
        "best_kl_loss": float(min(kl_losses)),
        "best_test_mse": float(min(test_mse_scores)),
        "best_test_psnr": float(max(test_psnr_scores)),
        "best_test_ssim": float(max(test_ssim_scores)),
        "total_losses": [float(x) for x in total_losses],
        "recon_losses": [float(x) for x in recon_losses],
        "kl_losses": [float(x) for x in kl_losses],
        "test_mse_scores": [float(x) for x in test_mse_scores],
        "test_psnr_scores": [float(x) for x in test_psnr_scores],
        "test_ssim_scores": [float(x) for x in test_ssim_scores],
        "beta": float(beta),
        "latent_dim": int(latent_dim),
    }

    metrics_path = os.path.join(BASE_DIR, "outputs", "metrics", "cvae_metrics.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Saved metrics to: {metrics_path}")

    return model, total_losses, recon_losses, kl_losses, test_mse_scores, test_psnr_scores, test_ssim_scores


if __name__ == "__main__":
    train_cvae(
        epochs=20,
        batch_size=50,
        learning_rate=0.001,
        num_filters=32,
        kernel_size=3,
        latent_dim=64,
        beta=0.5
    )