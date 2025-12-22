
import pandas as pd
import numpy as np

df = pd.read_csv('FTT-P_gamma.csv', header=None)

# duplicate first column into the next 100 columns
first = df.iloc[:, 0]
for i in range(1, 100):
    df[f'{df.columns[0]}_{i}'] = first.values

# set 4th row (index 3) to 2001..2100 (one value per column)
num_cols = df.shape[1]
vals = np.arange(2001, 2001 + num_cols)
df.iloc[0, :] = vals

# copy that 4th row to every 23rd row (4th, 27th, 50th, ...)
for r in range(0, len(df), 34):
    df.iloc[r, :] = df.iloc[0, :].values

empty_rows = pd.DataFrame([[''] * df.shape[1]] * 4, columns=df.columns)
df = pd.concat([empty_rows, df], ignore_index=True)

df.to_csv('gamma_value_formatted.csv', index=False, header=False)