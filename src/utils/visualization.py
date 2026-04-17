import os
import matplotlib.pyplot as plt
import numpy as np
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


def to_numpy(x):
    """
    Convert torch tensor to numpy array if needed.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x


def show_images(img_grey, img_real, img_fake, num_images=5, save_path="outputs/images/sample.png"):
    """
    Show grayscale input, ground truth, and predicted images.
    Optionally save the figure.
    """
    img_grey = to_numpy(img_grey)
    img_real = to_numpy(img_real)
    img_fake = to_numpy(img_fake)

    num_images = min(num_images, len(img_grey))

    # Convert (N, C, H, W) -> (N, H, W, C)
    img_grey = np.transpose(img_grey[:num_images], (0, 2, 3, 1)).squeeze(-1)
    img_real = np.transpose(img_real[:num_images], (0, 2, 3, 1))
    img_fake = np.transpose(img_fake[:num_images], (0, 2, 3, 1))

    plt.figure(figsize=(15, 8))

    for i in range(num_images):
        plt.subplot(3, num_images, i + 1)
        plt.imshow(img_grey[i], cmap="gray")
        plt.axis("off")
        if i == 0:
            plt.title("Input")

        plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(np.clip(img_real[i], 0, 1))
        plt.axis("off")
        if i == 0:
            plt.title("Ground Truth")

        plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(np.clip(img_fake[i], 0, 1))
        plt.axis("off")
        if i == 0:
            plt.title("Prediction")

    plt.tight_layout()

    if save_path:
        full_path = os.path.join(BASE_DIR, save_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        plt.savefig(full_path)
        print(f"Saved image to: {full_path}")

    plt.show()
    plt.close()


def plot_losses(train_losses, save_path=None):
    """
    Plot training loss curve.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, marker="o")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    if save_path:
        full_path = os.path.join(BASE_DIR, save_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        plt.savefig(full_path)
        print(f"Saved plot to: {full_path}")

    plt.show()
    plt.close()


def plot_cvae_losses(total_losses, recon_losses, kl_losses, save_path=None):
    """
    Plot CVAE losses.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(total_losses, label="Total Loss")
    plt.plot(recon_losses, label="Reconstruction Loss")
    plt.plot(kl_losses, label="KL Divergence")

    plt.title("CVAE Training Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    if save_path:
        full_path = os.path.join(BASE_DIR, save_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        plt.savefig(full_path)
        print(f"Saved plot to: {full_path}")

    plt.show()
    plt.close()


if __name__ == "__main__":
    img_grey = torch.rand(5, 1, 32, 32)
    img_real = torch.rand(5, 3, 32, 32)
    img_fake = torch.rand(5, 3, 32, 32)

    show_images(
        img_grey,
        img_real,
        img_fake,
        save_path="outputs/images/test.png"
    )

    losses = [0.9, 0.7, 0.5, 0.4]
    plot_losses(losses, save_path="outputs/plots/loss.png")