import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SensorWindowsDataset(Dataset):
    """
    PyTorch Dataset for sensor sliding windows with optional labels.
    """
    def __init__(self, windows_path: str, labels_path: str = None):
        """
        Args:
            windows_path (str): Path to .npy file containing windows, shape (N, seq_len, features).
            labels_path (str, optional): Path to .npy file containing labels, shape (N,) or (N,1).
        """
        self.windows = np.load(windows_path)  # e.g. imu_windows.npy
        if labels_path:
            self.labels = np.load(labels_path)
        else:
            self.labels = None

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx].astype(np.float32)
        if self.labels is not None:
            label = self.labels[idx].astype(np.float32)
            return torch.tensor(window), torch.tensor(label)
        else:
            return torch.tensor(window)

def get_dataloaders(windows_path, labels_path=None, batch_size=64, split_ratio=0.8, shuffle=True):
    """
    Creates training and validation DataLoaders from the dataset.

    Args:
        windows_path (str): Path to windows .npy file.
        labels_path (str, optional): Path to labels .npy file.
        batch_size (int): Batch size.
        split_ratio (float): Fraction of data to use for training.
        shuffle (bool): Whether to shuffle dataset before split.

    Returns:
        train_loader, val_loader (DataLoader, DataLoader): Train and validation loaders.
    """
    dataset = SensorWindowsDataset(windows_path, labels_path)
    dataset_size = len(dataset)
    train_size = int(split_ratio * dataset_size)
    val_size = dataset_size - train_size

    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    train_loader, val_loader = get_dataloaders("imu_windows.npy", batch_size=32)

    for batch_inputs in train_loader:
        print(batch_inputs.shape)  # Only input windows
        break
