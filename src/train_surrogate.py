import copy
import os

import mlflow
import numpy as np
import torch

from .heston import PARAM_RANGES

# === Helpers ===
def arrays_to_tensors(X: np.ndarray, Y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    '''Convert numpy arrays to PyTorch tensors.'''
    return torch.from_numpy(X.astype("float32")), torch.from_numpy(Y.astype("float32"))

def normalise_input(X:torch.Tensor, param_ranges: dict) -> torch.Tensor:
    '''Normalise input using min-max values from parameter ranges.'''
    mins = torch.tensor([bounds[0] for bounds in param_ranges.values()], dtype=X.dtype, device=X.device)
    maxs = torch.tensor([bounds[1] for bounds in param_ranges.values()], dtype=X.dtype, device=X.device)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    return (X - mins) / ranges


class OutputScaler:
    '''Z-score scaler for the vol surfaces. Fit on train only, then reused for
    val/test and at calibration time.'''
    def __init__(self, Y: torch.Tensor):
        self.mean = Y.mean(dim=0)
        self.std = Y.std(dim=0)
        self.std[self.std == 0] = 1.0

    def transform(self, Y: torch.Tensor) -> torch.Tensor:
        return (Y - self.mean) / self.std

    def inverse(self, Y: torch.Tensor) -> torch.Tensor:
        return Y * self.std + self.mean

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, d: dict) -> "OutputScaler":
        scaler = cls.__new__(cls)
        scaler.mean, scaler.std = d["mean"], d["std"]
        return scaler

# === Training Utilities ===
def train_model(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, num_epochs: int = 100, lr: float = 1e-3, patience: int = 10):
    '''Train model with Adam optimizer, MSELoss, validation monitoring, and early stopping.'''
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_state_dict = copy.deepcopy(model.state_dict())
    best_val_mse = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        # Training pass
        model.train()
        train_mse = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            Y_pred = model(X_batch)
            loss = criterion(Y_pred, Y_batch)
            loss.backward()
            optimizer.step()
            train_mse += loss.item() * X_batch.size(0)
        train_mse /= len(train_loader.dataset)
        
        # Validation pass
        model.eval()
        val_mse = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                Y_pred = model(X_batch)
                loss = criterion(Y_pred, Y_batch)
                val_mse += loss.item() * X_batch.size(0)
        val_mse /= len(val_loader.dataset)
        
        # Log to MLflow
        mlflow.log_metric("train_mse", train_mse, step=epoch)
        mlflow.log_metric("val_mse", val_mse, step=epoch)
        
        if epoch % 10 == 0:
            print(f"epoch {epoch:4d}  train_mse={train_mse:.6f}  val_mse={val_mse:.6f}")

        scheduler.step(val_mse)
        
        # Early stopping with best state tracking
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Restore best model
    model.load_state_dict(best_state_dict)
    return model

#=== Model ===
def create_model(input_dim: int = 5, output_dim: int = 63, hidden_dim: int = 64, num_layers: int = 4) -> torch.nn.Module:
    '''Feedforward net, num_layers counts Linear layers so the default is 3 hidden.
    ELU rather than ReLU because calibration differentiates this w.r.t. its inputs
    and a piecewise-linear surface makes that harder to descend.'''
    layers = []
    for i in range(num_layers):
        in_dim = input_dim if i == 0 else hidden_dim
        out_dim = output_dim if i == num_layers - 1 else hidden_dim
        layers.append(torch.nn.Linear(in_dim, out_dim))

        # No Activation for output
        if i < num_layers - 1:
            layers.append(torch.nn.ELU())

    return torch.nn.Sequential(*layers)

# === Dataset Class ===
def create_dataset(X: np.ndarray, Y: np.ndarray, param_ranges: dict,
                   y_scaler: "OutputScaler" = None) -> tuple[torch.utils.data.Dataset, "OutputScaler"]:
    '''Create a Dataset from numpy arrays. Pass the train set's scaler in for
    val/test so all three share one transform.'''
    dataset = SurrogateDataset(X, Y, param_ranges, y_scaler)
    return dataset, dataset.y_scaler

class SurrogateDataset(torch.utils.data.Dataset):
    '''PyTorch Dataset for Heston parameter sets and implied vol surfaces.'''
    def __init__(self, X: np.ndarray, Y: np.ndarray, param_ranges: dict,
                 y_scaler: "OutputScaler" = None):
        self.X, self.Y = arrays_to_tensors(X, Y)
        self.X = normalise_input(self.X, param_ranges)
        self.y_scaler = y_scaler if y_scaler is not None else OutputScaler(self.Y)
        self.Y = self.y_scaler.transform(self.Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# === Evaluation and Persistence ===
def evaluate(model: torch.nn.Module, X: np.ndarray, Y: np.ndarray, param_ranges: dict,
             y_scaler: OutputScaler) -> dict:
    '''Test error in implied vol points, so 0.01 is one vol point.'''
    model.eval()
    X_t, Y_t = arrays_to_tensors(X, Y)
    with torch.no_grad():
        pred = y_scaler.inverse(model(normalise_input(X_t, param_ranges)))
    err = (pred - Y_t).numpy()
    return {
        "rmse_vol_pts": float(np.sqrt((err ** 2).mean())),
        "mae_vol_pts": float(np.abs(err).mean()),
        "max_abs_err_vol_pts": float(np.abs(err).max()),
    }


def save_model(model: torch.nn.Module, y_scaler: OutputScaler, path: str = "models/vol_surrogate.pt"):
    '''Weights and scaler together; calibration needs the same transform.'''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "y_scaler": y_scaler.state_dict(),
        "input_dim": model[0].in_features,
        "output_dim": model[-1].out_features,
    }, path)


def load_model(path: str = "models/vol_surrogate.pt") -> tuple[torch.nn.Module, OutputScaler]:
    ckpt = torch.load(path, weights_only=False)
    model = create_model(input_dim=ckpt["input_dim"], output_dim=ckpt["output_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, OutputScaler.from_state_dict(ckpt["y_scaler"])


def run_training(data_path: str = "data/heston_surfaces.npz", epochs: int = 300,
                 lr: float = 1e-3, batch_size: int = 64, hidden_dim: int = 64,
                 patience: int = 25, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    data = np.load(data_path)

    train_set, y_scaler = create_dataset(data["X_train"], data["Y_train"], PARAM_RANGES)
    val_set, _ = create_dataset(data["X_val"], data["Y_val"], PARAM_RANGES, y_scaler)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)

    model = create_model(input_dim=data["X_train"].shape[1],
                         output_dim=data["Y_train"].shape[1], hidden_dim=hidden_dim)

    mlflow.set_experiment("heston-surrogate")
    with mlflow.start_run():
        mlflow.log_params({"epochs": epochs, "lr": lr, "batch_size": batch_size,
                           "hidden_dim": hidden_dim, "patience": patience, "seed": seed,
                           "n_train": len(train_set), "n_outputs": data["Y_train"].shape[1]})
        model = train_model(model, train_loader, val_loader, num_epochs=epochs,
                            lr=lr, patience=patience)
        metrics = evaluate(model, data["X_test"], data["Y_test"], PARAM_RANGES, y_scaler)
        mlflow.log_metrics(metrics)

    save_model(model, y_scaler)
    return metrics


if __name__ == "__main__":
    metrics = run_training()
    print("\nTest set (implied vol points, 0.01 = one vol point):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.5f}")
