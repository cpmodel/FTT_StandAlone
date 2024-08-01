# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 13:48:43 2024

@author: rs1132
"""

import matplotlib.pyplot as plt
import pandas as pd

# Replace 'path_to_your_file.csv' with the actual path to your CSV file
file_path = 'crossover_years_plot.csv'

# Reading the data from the CSV file
df = pd.read_csv(file_path)

# Assuming the first column in your CSV file is the index (e.g., 'Policies')
df.set_index(df.columns[0], df.rows[1], inplace=True)

# Plotting the table
fig, ax = plt.subplots(figsize=(10, 2))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

plt.title("Cost Parity Brought Forward by (months)", pad=20)
plt.show()
