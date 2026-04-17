import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import json

# Add project root to path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data.loader import load_cifar10, get_batch
from src.data.preprocess import prepare_colourization_data
from src.models.cnn import RegressionCNN
from src.utils.visualization import show_images, plot_losses
from src.utils.metrics import mse_metric, psnr, ssim


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


def evaluate_model(model, test_grey, test_rgb, batch_size, device):
    """
    Evaluate model on test set using MSE, PSNR, and SSIM.
    """
    model.eval()

    mse_scores = []
    psnr_scores = []
    ssim_scores = []

    with torch.no_grad():
        for batch_x, batch_y in get_batch(test_grey, test_rgb, batch_size):
            batch_x = torch.tensor(batch_x, dtype=torch.float32).to(device)
            batch_y = torch.tensor(batch_y, dtype=torch.float32).to(device)

            outputs = model(batch_x)

            mse_scores.append(mse_metric(outputs, batch_y))
            psnr_scores.append(psnr(outputs, batch_y))
            ssim_scores.append(ssim(outputs, batch_y))

    return np.mean(mse_scores), np.mean(psnr_scores), np.mean(ssim_scores)


def train_cnn(
    epochs=20,
    batch_size=100,
    learning_rate=0.001,
    num_filters=32,
    kernel_size=3,
    device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device:", device)

    # Load data
    print("Loading CIFAR-10...")
    (x_train, y_train), (x_test, y_test) = load_cifar10()

    print("Preprocessing horse images...")
    train_rgb, train_grey = prepare_colourization_data(x_train, y_train)
    test_rgb, test_grey = prepare_colourization_data(x_test, y_test)

    print("Train RGB shape:", train_rgb.shape)
    print("Train Grey shape:", train_grey.shape)
    print("Test RGB shape:", test_rgb.shape)
    print("Test Grey shape:", test_grey.shape)

    # Model
    model = RegressionCNN(kernel_size=kernel_size, num_filters=num_filters).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    test_mse_scores = []
    test_psnr_scores = []
    test_ssim_scores = []

    best_loss = float("inf")

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        batch_losses = []

        for batch_x, batch_y in get_batch(train_grey, train_rgb, batch_size):
            batch_x = torch.tensor(batch_x, dtype=torch.float32).to(device)
            batch_y = torch.tensor(batch_y, dtype=torch.float32).to(device)

            optimizer.zero_grad()

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        epoch_loss = np.mean(batch_losses)
        train_losses.append(epoch_loss)

        # Evaluate on test set
        test_mse, test_psnr, test_ssim = evaluate_model(
            model, test_grey, test_rgb, batch_size, device
        )
        test_mse_scores.append(test_mse)
        test_psnr_scores.append(test_psnr)
        test_ssim_scores.append(test_ssim)

        print(
            f"Epoch [{epoch+1}/{epochs}] - "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Test MSE: {test_mse:.4f} | "
            f"Test PSNR: {test_psnr:.4f} | "
            f"Test SSIM: {test_ssim:.4f}"
        )

        metrics = {
    "best_train_loss": float(min(train_losses)),
    "best_test_mse": float(min(test_mse_scores)),
    "best_test_psnr": float(max(test_psnr_scores)),
    "best_test_ssim": float(max(test_ssim_scores)),
}

        metrics_path = os.path.join(BASE_DIR, "outputs", "metrics", "cnn_metrics.json")
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Saved metrics to: {metrics_path}")

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            model_path = os.path.join(BASE_DIR, "outputs", "models", "cnn_best.pth")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to: {model_path}")

        # Save sample predictions every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                sample_x = torch.tensor(test_grey[:5], dtype=torch.float32).to(device)
                sample_y = torch.tensor(test_rgb[:5], dtype=torch.float32).to(device)
                sample_pred = model(sample_x)

                show_images(
                    sample_x,
                    sample_y,
                    sample_pred,
                    num_images=5,
                    save_path=f"outputs/images/cnn_epoch_{epoch+1}.png"
                )

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")

    # Save final loss curve
    plot_losses(train_losses, save_path="outputs/plots/cnn_loss_curve.png")

    print("\nFinal Metrics:")
    print(f"Best Train Loss: {min(train_losses):.4f}")
    print(f"Best Test MSE: {min(test_mse_scores):.4f}")
    print(f"Best Test PSNR: {max(test_psnr_scores):.4f}")
    print(f"Best Test SSIM: {max(test_ssim_scores):.4f}")

    return model, train_losses, test_mse_scores, test_psnr_scores, test_ssim_scores


if __name__ == "__main__":
    train_cnn(
        epochs=20,
        batch_size=100,
        learning_rate=0.001,
        num_filters=32,
        kernel_size=3
    )