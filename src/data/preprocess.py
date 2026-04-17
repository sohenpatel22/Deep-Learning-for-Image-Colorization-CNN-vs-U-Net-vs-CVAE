import numpy as np
import torch
import torch.nn as nn

# CIFAR-10 class index for horse
HORSE_CLASS = 7


def normalize_images(x, max_pixel=255.0):
    """
    Normalize image pixel values to [0, 1].
    """
    return x.astype(np.float32) / max_pixel


def filter_by_class(x, y, class_label=HORSE_CLASS):
    """
    Keep only images belonging to one class.
    """
    indices = np.where(y == class_label)[0]
    return x[indices], y[indices]


def rgb_to_grayscale(x):
    """
    Convert RGB images to grayscale by taking mean across channels.

    Input shape:  (N, 3, 32, 32)
    Output shape: (N, 1, 32, 32)
    """
    grey = np.mean(x, axis=1, keepdims=True)
    return grey.astype(np.float32)


def downsize_and_upsize(x):
    """
    Downsize and then upsize images.
    This can be used for the optional low-resolution input experiment.
    """
    downsize_module = nn.Sequential(
        nn.AvgPool2d(2),
        nn.AvgPool2d(2),
        nn.Upsample(scale_factor=2),
        nn.Upsample(scale_factor=2),
    )

    x_tensor = torch.from_numpy(x).float()
    x_out = downsize_module(x_tensor)
    return x_out.numpy()


def prepare_colourization_data(x, y, class_label=HORSE_CLASS, downsize_input=False):
    """
    Full preprocessing pipeline for colourization project.

    Steps:
    1. Keep only selected class
    2. Normalize RGB images
    3. Convert RGB images to grayscale

    Returns:
        rgb_images  -> shape (N, 3, 32, 32)
        grey_images -> shape (N, 1, 32, 32)

    If downsize_input=True:
        returns rgb_images and downsized_rgb_images instead
    """
    x, y = filter_by_class(x, y, class_label=class_label)
    x = normalize_images(x)

    # shuffle data
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    x = x[indices]

    if downsize_input:
        x_downsized = downsize_and_upsize(x)
        return x, x_downsized

    grey = rgb_to_grayscale(x)
    return x, grey


if __name__ == "__main__":
    # small dummy test
    x_dummy = np.random.randint(0, 256, size=(10, 3, 32, 32), dtype=np.uint8)
    y_dummy = np.array([7, 1, 7, 3, 7, 7, 2, 7, 0, 7])

    rgb, grey = prepare_colourization_data(x_dummy, y_dummy)

    print("RGB shape:", rgb.shape)
    print("Grey shape:", grey.shape)
    print("RGB min/max:", rgb.min(), rgb.max())
    print("Grey min/max:", grey.min(), grey.max())