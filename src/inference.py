import torch
import numpy as np
import cantera as ct
from argparse import ArgumentParser 
from architecture import IgnitionNet
from utils import StandardScaler, model_inference, compute_adiabatic_flame_temperature, squared_err, abs_err

fuel      = "NC10H22:0.42,C6H6:0.15,cC6H12:0.43"
oxidizer  = "O2:1,N2:3.76"
mechanism = "../jetsurf2.0/jetsurf2.0.yaml" 

features     = ["t0", "p_atm", "phi"]
target       = "adiabatic_flame_temperature"
input_dims   = len(features)
output_dims  = 1
scaler       = StandardScaler.load("scaler.npz")

parser = ArgumentParser(description = "IgnitionNet Inference")
parser.add_argument("--t0",         type = float, required = True, help = "Initial Temperature (K)")
parser.add_argument("--p_atm",      type = float, required = True, help = "Pressure (atm)")
parser.add_argument("--phi",        type = float, default  = 1.0,  help = "Equivalence Ratio")
parser.add_argument("--checkpoint", type = str,   default = "ignition_node.pt", help = "Path To Saved Model Checkpoint")
parser.add_argument("--validation", action = "store_true", help = "Whether To Validate The Model's Predictions By Running The Cantera Simulation After")
args = parser.parse_args()

# model  = IgnitionNet(input_dims = input_dims, output_dims = output_dims)
# model.load_state_dict(torch.load(args.checkpoint))
model  = IgnitionNet.load(args.checkpoint)

# predict the adiabatic flame tempearture
pred = model_inference(
    model      = model, 
    scaler     = scaler, 
    t0         = args.t0, 
    p_atm      = args.p_atm, 
    phi        = args.phi,
    validation = args.validation,
    fuel       = fuel,
    oxidizer   = oxidizer,
    mechanism  = mechanism,
)

print("========== INPUTS ==========")
print(f" t0        = {args.t0} K")
print(f" p_atm     = {args.p_atm}    ATM")
print(f" phi       = {args.phi}")
print("========== RESULTS ==========")
print(f" t_ad      = {round(pred, 2)} K")
if args.validation:
    t_corr = compute_adiabatic_flame_temperature(
        t0        = args.t0,
        p_atm     = args.p_atm,
        phi       = args.phi,
        fuel      = fuel, 
        oxidizer  = oxidizer,
        mechanism = mechanism,
    )
    t_sq_err  = squared_err(t_corr, pred)
    t_abs_err = abs_err(t_corr, pred)
    print(f" t_corr    = {round(t_corr,    2)} K")
    print(f" t_sq_err  = {round(t_sq_err,  2)} K")
    print(f" t_abs_err = {round(t_abs_err, 2)}   K")
print("=============================")

# === EXAMPLE NO.1 === # 
# python inference.py --t0 400.0 \
# --p_atm 5.0 \
# --phi 1.75 \
# --checkpoint ../models/00e79fa0-23cf-47af-b395-846135eda8fb/model.pt \
# --validation

# === EXAMPLE NO.2 === # 
# python inference.py  --t0 400.0 \
# --p_atm 5.0 \
# --phi 1.75 \
# --checkpoint ../models/f5e01293-24aa-4581-bc0c-445ebc8fc1bb/model.pt \
# --validation

# === EXAMPLE NO.3 === # 
# python inference.py --t0 400.0 \
# --p_atm 5.0 \
# --phi 1.75 \
# --checkpoint ../models/993e8c34-ce99-4e62-952f-2cf36655c6ce/model.pt \
# --validation

# === EXAMPLE NO.4 === # 
# python inference.py --t0 400.0 \
# --p_atm 5.0 \
# --phi 1.75 \
# --checkpoint ../models/054dc26b-8a6f-40ec-b49b-aef99d5ea799/model.pt \
# --validation

# === EXAMPLE NO.5 === # 
# python inference.py --t0 400.0 \
# --p_atm 5.0 \
# --phi 1.75 \
# --checkpoint ../models/da657878-adf1-4771-b4e2-3239a6c2224d/model.pt \
# --validation
