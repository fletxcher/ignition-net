import os
import time
import torch
import itertools
import functools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from architecture import IgnitionNet
from utils import StandardScaler, model_inference, compute_adiabatic_flame_temperature

fuel            = "NC10H22:0.42,C6H6:0.15,cC6H12:0.43"
oxidizer        = "O2:1,N2:3.76"
mechanism       = "../jetsurf2.0/jetsurf2.0.yaml" 
features        = ["t0", "p_atm", "phi"]
target          = "adiabatic_flame_temperature"
input_dims      = len(features)
output_dims     = 1
model           = IgnitionNet.load(
                        input_dims = input_dims,
                        output_dims = output_dims,
                        path = os.path.join(
                            os.path.dirname(__file__),
                            "..",
                            "models",
                            "da657878-adf1-4771-b4e2-3239a6c2224d",
                            "model.pt"
                        )
                    )
scaler          = StandardScaler.load("scaler.npz")
temperatures    = np.linspace(400.0, 1000.0, num = 3)
pressures       = np.linspace(5.0,   40.0,   num = 3)
phis            = np.linspace(0.5,   2.0,    num = 3)

surrogate_times     = []
surrogate_preds     = []
cantera_times       = []
cantera_solutions   = []

surrogate_fn = functools.partial(
                    model_inference, 
                    model, 
                    scaler
                )
cantera_fn = functools.partial(
                    compute_adiabatic_flame_temperature, 
                    fuel      = fuel, 
                    oxidizer  = oxidizer, 
                    mechanism = mechanism
                )
n_combinations = len(temperatures) * len(pressures) * len(phis)
variables = itertools.product(temperatures, pressures, phis)
for args in tqdm(variables, total = n_combinations):
    t0 = time.perf_counter()
    # surrogate_fn(*args)
    # surrogate_times.append((time.perf_counter() - t0) * 1000)
    s_pred = surrogate_fn(*args)
    surrogate_times.append((time.perf_counter() - t0) * 1000)
    surrogate_preds.append(s_pred)
    
    t0 = time.perf_counter()
    # cantera_fn(*args)
    # cantera_times.append((time.perf_counter() - t0) * 1000)
    c_pred = cantera_fn(*args)
    cantera_times.append((time.perf_counter() - t0) * 1000)
    cantera_solutions.append(c_pred)

x = range(n_combinations)
fig = plt.figure(figsize = (16, 8))
# === EXECUTION TIME === #
plt.plot(x, cantera_times  , color = "red",  label = "Cantera",     linestyle = "-")
plt.plot(x, surrogate_times, color = "blue", label = "Ignition-Net", linestyle = "--")
plt.fill_between(x, surrogate_times, cantera_times, alpha = 0.15, color = "green")
plt.title("Surrogate Model Vs Cantera Simulations Inference Speed")
plt.xlabel("Trials")
plt.ylabel("Time (ms)")
plt.grid(visible = True, alpha = 0.3)
plt.legend()
plt.tight_layout()
plt.savefig("speed.png")

# === ERROR === #
errors = np.abs(np.array(surrogate_preds) - np.array(cantera_solutions))
fig = plt.figure(figsize = (16, 8))
plt.plot(x, errors, color = "red", label = "Absolute Error", linestyle = "-")
plt.axhline(np.median(errors), color = "black", linestyle = "--", lw = 1.2, label = f"Median: {np.median(errors):.2f} K")
plt.fill_between(x, 0, errors, alpha = 0.15, color = "red")
plt.title("Surrogate Model Prediction Error")
plt.xlabel("Trials")
plt.ylabel("Absolute Error (K)")
plt.legend()
plt.grid(visible = True, alpha=0.3)
plt.margins(0)
plt.tight_layout()
plt.savefig("error.png", dpi = 150)

# === EXECUTION TIME === #
# ax1.plot(x, cantera_times,   color = "red",  label = "Cantera",      linestyle = "-")
# ax1.plot(x, surrogate_times, color = "blue", label = "Ignition-Net", linestyle = "--")
# ax1.fill_between(x, surrogate_times, cantera_times, alpha = 0.15, color = "green")
# ax1.set_title("Inference Speed")
# ax1.set_xlabel("Trials")
# ax1.set_ylabel("Time (ms)")
# ax1.legend()
# ax1.grid(visible = True, alpha = 0.3)
# ax1.margins(0)