# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 15:31:49 2024

@author: Owner
"""

import matplotlib.pyplot as plt
import pandas as pd

file_path = 'crossover_years_plot.csv'

df = pd.read_csv(file_path, skiprows=1)

df.set_index(df.columns[0], inplace=True)

# Converting each value from years to months
df = df * 12

# Rounding numbers to two decimal places
df = df.round(2)

# Plotting the table
fig, ax = plt.subplots(figsize=(12, 4))

ax.text(0.5, 0.8, 'Policies', ha='center', va='center', transform=ax.transAxes, fontsize=14, fontweight='bold')
ax.text(-0.1, 0.5, 'Sectors', ha='center', va='center', rotation='vertical', transform=ax.transAxes, fontsize=14, fontweight='bold')

ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center', edges='BRLT')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.2)

header_color = '#40466e'
header_text_color = 'w'
row_colors = ['#f2f2f2', 'w']

for (i, j), cell in table.get_celld().items():
    if j == -1:
        cell.set_text_props(fontweight='bold', color='black')  # Row labels bold
    if i == 0:
        cell.set_facecolor(header_color)
        cell.set_text_props(color=header_text_color, fontweight='bold')  # Column labels white text and bold
    if i > 0 and j > -1:
        cell.set_facecolor(row_colors[i % 2])  # Alternating row colors

plt.title("Cost Parity Brought Forward by (months)", pad=10, fontsize=16, fontweight='bold')

plt.show()
