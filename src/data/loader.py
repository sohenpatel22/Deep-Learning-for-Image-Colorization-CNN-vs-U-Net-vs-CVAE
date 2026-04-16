import os
import pickle
import tarfile
import numpy as np
from urllib.request import urlretrieve

# CIFAR-10 download URL
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def download_cifar10(data_dir="data"):
    """
    Downloads and extracts CIFAR-10 dataset if not already present.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    tar_path = os.path.join(data_dir, "cifar-10-python.tar.gz")
    extract_path = os.path.join(data_dir, "cifar-10-batches-py")

    # Download
    if not os.path.exists(tar_path):
        print("Downloading CIFAR-10")
        urlretrieve(CIFAR_URL, tar_path)

    # Extract
    if not os.path.exists(extract_path):
        print("Extracting CIFAR-10")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=data_dir)

    return extract_path


def load_batch(file_path):
    """
    Loads a single CIFAR-10 batch file.
    """
    with open(file_path, "rb") as f:
        data_dict = pickle.load(f, encoding="bytes")

    data = data_dict[b"data"]
    labels = data_dict[b"labels"]

    # reshape to (N, 3, 32, 32)
    data = data.reshape(-1, 3, 32, 32)
    labels = np.array(labels)

    return data, labels


def load_cifar10(data_dir="data"):
    """
    Loads full CIFAR-10 dataset.
    Returns:
        (x_train, y_train), (x_test, y_test)
    """
    path = download_cifar10(data_dir)

    x_train = []
    y_train = []

    # Load training batches (1 to 5)
    for i in range(1, 6):
        file_path = os.path.join(path, f"data_batch_{i}")
        data, labels = load_batch(file_path)
        x_train.append(data)
        y_train.append(labels)

    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    # Load test batch
    x_test, y_test = load_batch(os.path.join(path, "test_batch"))

    return (x_train, y_train), (x_test, y_test)


def get_batch(x, y, batch_size):
    """
    Simple batch generator.
    """
    n = x.shape[0]

    for i in range(0, n, batch_size):
        batch_x = x[i:i + batch_size]
        batch_y = y[i:i + batch_size]
        yield batch_x, batch_y


if __name__ == "__main__":
    # Quick test
    (x_train, y_train), (x_test, y_test) = load_cifar10()

    print("Train shape:", x_train.shape)
    print("Test shape:", x_test.shape)