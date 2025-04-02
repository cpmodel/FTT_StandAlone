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

df = df[df["Gamma"] != "Gamma"]
lines_per_r = int(len(df)/71)

for r in range(71):
    country_data = df.iloc[lines_per_r*r:lines_per_r*(r+1)]
    car_data = country_data.iloc[:27]
    reshaped_data = np.array(car_data).reshape(9, 3, -1)  # 9 groups, each with 3 rows
    data_float = np.asarray(reshaped_data, dtype=float)
    df_non_zero = data_float[(data_float != 0).all(axis=1)[:,0]]
    df_average = np.mean(df_non_zero, axis=0, keepdims=True)
    df_average_broadcasted = np.tile(df_average, (9, 1))
