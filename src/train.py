import os
import uuid
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from typing import List
from architecture import IgnitionNet 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import StandardScaler, build_loss_curves, compute_metrics

class IgnitionNetDataset(Dataset):
    def __init__(self, csv_path: str, features: List[str], target: str):
        df = pd.read_csv(csv_path)
        df.drop(columns = ["Unnamed: 0"], inplace = True)
        scaler = StandardScaler()
        scaler.fit_transform(df[features].to_numpy())
        scaler.save(path = "scaler.npz")
        self.x  = scaler.fit_transform(df[features].to_numpy())
        self.y  = df[target].to_numpy()
    
    def __getitem__(self, index) -> dict:
        sample = {
            "features": torch.tensor(
                self.x[index],
                dtype = torch.float32 
            ),
            "targets": torch.tensor(
                self.y[index], 
                dtype = torch.float32 
            ).flatten(),
        }
        return sample
    
    def __len__(self) -> int:
        return len(self.x)

n_epochs     = 100
eta          = 1e-1
device       = "cuda:0"
csv_path     = "../datasets/data1.csv"
features     = ["t0", "p_atm", "phi"]
target       = "adiabatic_flame_temperature"
input_dims   = len(features)
output_dims  = 1
session_id   = uuid.uuid4().hex
base_path    = f"../models/{session_id}"
model_path   = f"{base_path}/model.pt"
plot_path    = f"{base_path}/losses.png"
train_losses = []
val_losses   = []

# make directory for unique training session
os.makedirs(
    base_path,
    exist_ok = True
)

# configure a logger to write to a log file to help with monitoring training and validation losses
logging.basicConfig(
    filename = f"{base_path}/training.log", 
    level    = logging.INFO,
    format   = "%(asctime)s - %(levelname)s - %(message)s",
    datefmt  = "%Y-%m-%d %H:%M:%S"
)

# construct the model, optimizer, and loss function
model     = IgnitionNet(input_dims = input_dims, output_dims = output_dims)
optimizer = torch.optim.Adam(params = model.parameters(), lr = eta)
loss_fn   = nn.MSELoss()

# create a custom dataset object so that we can take advantage of torch's random_split 
dataset = IgnitionNetDataset(
    csv_path = csv_path,
    features = features,
    target   = target 
)

# 70/20/10 split for training, testing, and validation 
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [0.7, 0.2, 0.1]
)
# create data loaders
train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
val_loader   = DataLoader(val_dataset,   batch_size = 64, shuffle = True)
test_loader  = DataLoader(test_dataset,  batch_size = 64, shuffle = True)

for epoch in tqdm(range(n_epochs)):
    logging.info(f"======== Epoch: {epoch + 1} ========")
    
    model.train()
    train_loss = 0.0
    for data in train_loader:
        inputs, targets = data["features"], data["targets"]
        optimizer.zero_grad()

        # make predictions
        predictions = model(inputs)

        # compute the loss and its gradients 
        loss = loss_fn(predictions, targets)
        loss.backward()
        
        # update the value of train_loss
        train_loss += loss.item()

        # adjust learning weights
        optimizer.step()

    train_loss /= len(train_loader.dataset)
    logging.info(f"Avg Train Loss: {round(train_loss, 2)}")
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            inputs, targets = data["features"], data["targets"]
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            val_loss += loss.item() 

        val_loss /= len(val_loader.dataset)
        logging.info(f"Avg Val Loss:   {round(val_loss, 2)}")
        val_losses.append(val_loss)

    # implement early stopping or convergence checking ? 

# overall summary of the training, along with some metrics for evaluation performance before moving to inference
r2, mse, nrmse, mae = compute_metrics(model = model, dataloader = test_loader)
avg_train_loss = np.mean(train_losses)
avg_val_loss = np.mean(val_losses)
logging.info("========== SUMMARY =========")
logging.info(f"Avg Train Loss: {round(avg_train_loss, 2)}")
logging.info(f"Avg Val Loss:   {round(avg_val_loss,   2)}")
logging.info(f"R^2 Score:      {round(r2,             2)}")
logging.info(f"MSE Score:      {round(mse,            2)}")
logging.info(f"NRMSE Score:    {round(nrmse,          2)}")
logging.info(f"MAE Score:      {round(nrmse,          2)}")
logging.info("============================")

# save the final model and the plot of the training and validation curves vs epochs
torch.save(model.state_dict(), model_path)
build_loss_curves(
    n_epochs     = n_epochs,
    train_losses = train_losses,  
    val_losses   = val_losses,  
    path         = plot_path,  
)
    