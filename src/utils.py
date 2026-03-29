import torch
import numpy as np
import cantera as ct
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import List, Self, Tuple, Callable
from torchmetrics import R2Score, MeanSquaredError, NormalizedRootMeanSquaredError, MeanAbsoluteError

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray): 
        self.mean  = X.mean(axis = 0)
        self.std   = X.std(axis = 0)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str) -> None:
        np.savez(path, mean = self.mean, std = self.std)

    @classmethod
    def load(cls, path: str) -> Self:
        data         = np.load(path, allow_pickle = True)
        scaler       = cls()
        scaler.mean  = data["mean"]
        scaler.std   = data["std"]
        return scaler

def compute_adiabatic_flame_temperature(
        t0: float, 
        p_atm: float, 
        phi: float, 
        fuel: str, 
        oxidizer: str, 
        mechanism: str
    ) -> float:
    """
    Computes the adiabatic flame temperature
    """
    gas    = ct.Solution(mechanism)
    gas.TP = t0, p_atm * ct.one_atm
    gas.set_equivalence_ratio(
        phi      = phi, 
        fuel     = fuel, 
        oxidizer = oxidizer
    )
    gas.equilibrate("HP")
    return gas.T

def squared_err(y: float, y_pred: float) -> float:
    """
    Computes the squared error between y and y_pred
    """
    return (y - y_pred) ** 2 

def abs_err(y: float, y_pred: float) -> float:
    """
    Computes the absolute error between y and y_pred
    """
    return abs(y - y_pred)

def build_loss_curves(
        n_epochs: int, 
        train_losses: List[float], 
        val_losses: List[float], 
        path: str
    ) -> None:
    """
    Generate plot of the training and validation loss curves
    """
    fig  = plt.figure(figsize = (16, 8))
    epochs = [i for i in range(n_epochs)]
    plt.title("Training Vs Validation Loss Curves")
    plt.plot(epochs, train_losses, color = "blue", label = "Training Loss", linestyle = "-")
    plt.plot(epochs, val_losses, color = "red", label = "Validation Loss", linestyle = "--")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(visible = True, alpha = 0.5)
    plt.legend()
    plt.savefig(path)

def compute_metrics(
        model: nn.Module, 
        dataloader: DataLoader
    ) -> float:
    """
    Computes the R^2, MSE, NRMSE, and MAE scores to measure model performance
    """
    r2_score = R2Score()
    mse_score = MeanSquaredError()
    nrmse_score = NormalizedRootMeanSquaredError()
    mae_score = MeanAbsoluteError()
    with torch.no_grad():
        for data in dataloader:
            inputs, targets = data["features"], data["targets"]
            predictions = model(inputs)
            r2_score.update(predictions, targets)
            mse_score.update(predictions, targets)
            nrmse_score.update(predictions, targets)
            mae_score.update(predictions, targets)
    r2 = r2_score.compute().item()
    mse = mse_score.compute().item()
    nrmse = nrmse_score.compute().item()
    mae = mae_score.compute().item()
    return r2, mse, nrmse, mae

def model_inference(
        model: nn.Module, 
        scaler: StandardScaler, 
        t0: float, 
        p_atm: float, 
        phi: float,
    ) -> float:
    """
    Predicts the adiabatic flame temperature given t0, p_atm, and phi
    """
    # set the module in evaluation mode.
    model.eval()

    # transform the inputs using the same scaler
    scaled = scaler.transform(np.array([t0, p_atm, phi]))
    inputs = torch.tensor(scaled, dtype = torch.float32)

    pred = None
    # disable tracking of gradients in autograd
    with torch.no_grad():
        pred = model(inputs).item()
    return pred

    