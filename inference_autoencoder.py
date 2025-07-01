import torch
import torch.nn.functional as F
import numpy as np
from maintenance_model import SensorAutoencoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_autoencoder_model(model_path):
    model = SensorAutoencoder().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

def compute_reconstruction_error(model, window_tensor):
    """
    Takes a single window or batch of windows and computes MSE reconstruction error.
    Input shape: (window_size, num_features) or (batch_size, window_size, num_features)
    """
    if isinstance(window_tensor, np.ndarray):
        window_tensor = torch.tensor(window_tensor, dtype=torch.float32)

    if window_tensor.ndim == 2:
        window_tensor = window_tensor.unsqueeze(0)  # Make it (1, window, features)

    window_tensor = window_tensor.to(DEVICE)
    with torch.no_grad():
        reconstructed = model(window_tensor)
        mse = F.mse_loss(reconstructed, window_tensor, reduction='none')
        mse_per_sample = mse.view(mse.shape[0], -1).mean(dim=1)  # mean per window

    return mse_per_sample.cpu().numpy()  # returns a numpy array of shape (batch_size,)

def is_anomalous(reconstruction_error, threshold=0.05):
    """
    Simple thresholding function. You’ll likely want to tune this.
    """
    return reconstruction_error > threshold