# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 15:00:14 2024

@author: rs1132
"""

import openpyxl

# Load the workbook and select the sheet
workbook = openpyxl.load_workbook('FTT-Fr-20x71_2022_S0.xlsx')
sheet = workbook['ZCET']

# List of cells to copy from
source_cells = ['C11', 'C19', 'C25']

# Destination starting row
start_row = 11
destination_column = 'C'

# Number of iterations
iterations = 32

# Loop through each iteration
for i in range(iterations):
    # Calculate the starting row for this iteration
    iteration_start_row = start_row + i * 21
    
    # Copy each cell to the corresponding row in the iteration
    for cell in source_cells:
        source_value = sheet[cell].value
        
        # Calculate the destination row based on the iteration start row
        original_row = int(cell[1:])
        destination_row = iteration_start_row + (original_row - start_row)
        
        # Determine the destination cell
        destination_cell = f'{destination_column}{destination_row}'
        
        # Copy the value to the destination cell
        sheet[destination_cell] = source_value

# Save the workbook
workbook.save('FTT-Fr-20x71_2022_S0.xlsx')
