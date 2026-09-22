import torch
import pandas as pd
import numpy as np

# === Helpers ===
def pandas_to_tensors(X: pd.DataFrame, Y: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    '''Convert pandas DataFrames to PyTorch tensors.'''
    return torch.from_numpy(X.values.astype("float32")), torch.from_numpy(Y.values.astype("float32"))

def normalise_input(X:torch.Tensor, param_ranges: dict) -> torch.Tensor:
    '''Normalise input using min-max values from parameter ranges.'''
    mins = torch.tensor([bounds[0] for bounds in param_ranges.values()], dtype=X.dtype, device=X.device)
    maxs = torch.tensor([bounds[1] for bounds in param_ranges.values()], dtype=X.dtype, device=X.device)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    return (X - mins) / ranges

# === Dataset Class ===
class SurrogateDataset(torch.utils.data.Dataset):
    '''PyTorch Dataset for Heston parameter sets and implied vol surfaces.'''
    def __init__(self, X: pd.DataFrame, Y: pd.DataFrame, param_ranges: dict):
        self.X, self.Y = pandas_to_tensors(X, Y)
        self.X = normalise_input(self.X, param_ranges)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


data = np.load("data/heston_surfaces.npz")
print(data.files)

print("X_train shape:", data["X_train"].shape)
print("X_test shape:", data["X_test"].shape)
print("X_val shape:", data["X_val"].shape)
print("Y_train shape:", data["Y_train"].shape)
print("Y_test shape:", data["Y_test"].shape)
print("Y_val shape:", data["Y_val"].shape)
print("moneyness_grid:", data["moneyness_grid"])
print("maturity_grid:", data["maturity_grid"])
