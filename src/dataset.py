import torch
import itertools
import numpy as np
import pandas as pd
import cantera as ct
from tqdm import tqdm
# import matplotlib.pyplot as plt
from typing import List
from torch.utils.data import Dataset, random_split
from utils import compute_adiabatic_flame_temperature

fuel      = "NC10H22:0.42,C6H6:0.15,cC6H12:0.43"
oxidizer  = "O2:1,N2:3.76"
mechanism = "../jetsurf2.0/jetsurf2.0.yaml" 

temperatures = np.linspace(400.0, 1000.0, num = 25)
pressures    = np.linspace(5.0,   40.0,   num = 25)
phis         = np.linspace(0.5,   2.0,    num = 25)

data = []

n_combinations = len(temperatures) * len(pressures) * len(phis)
variables = itertools.product(temperatures, pressures, phis)
for t0, p_atm, phi in tqdm(variables, total = n_combinations):
    adiabatic_flame_temperature = compute_adiabatic_flame_temperature(
        t0        = t0, 
        p_atm     = p_atm, 
        phi       = phi, 
        fuel      = fuel, 
        oxidizer  = oxidizer, 
        mechanism = mechanism
    )
    data.append({
        "t0": t0,
        "p_atm": p_atm,
        "phi": phi,
        "adiabatic_flame_temperature": adiabatic_flame_temperature
    })

df = pd.DataFrame(data = data)
df.to_csv("../datasets1/data1.csv")


