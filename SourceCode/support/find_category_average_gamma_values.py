# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 13:16:11 2025

@author: Femke Nijsse
"""

from pathlib import Path
import pandas as pd
import numpy as np

file_path = Path(__file__).resolve().parents[2] / "FTT-Tr_gamma.csv"
df = pd.read_csv(file_path)

#df = df[df["Gamma"] != "Gamma"]
new_row = pd.DataFrame([['Gamma'] * df.shape[1]], columns=df.columns)
df = pd.concat([new_row, df], ignore_index=True)
lines_per_r = int(len(df)/71)


for r in range(2):
    country_data = df.iloc[lines_per_r * r + 1 : lines_per_r * (r+1)]
    car_data = country_data.iloc[:27]
    reshaped_data = np.array(car_data).reshape(9, 3, -1)  # 9 groups, each with 3 rows
    data_float = np.asarray(reshaped_data, dtype=float)
    data_non_zero = data_float[(data_float != 0).all(axis=1)[:,0]]
    data_average = np.mean(data_non_zero, axis=0, keepdims=True)
    data_average_broadcasted = np.tile(data_average, (9, 1))[0, :, 0]
    data_average_broadcasted = np.hstack([data_average_broadcasted, np.zeros(8)])[:, np.newaxis]
    df.iloc[lines_per_r * r + 1 : lines_per_r * (r+1)] = data_average_broadcasted
    
# Print df back into csv

# %% Saving almost to the right format (I'm naming the gamma row the same for each model.. )
df = df.map(lambda x: np.round(x, 2) if isinstance(x, (int, float)) else x)
df.to_csv("FTT-Tr tech-average.csv")