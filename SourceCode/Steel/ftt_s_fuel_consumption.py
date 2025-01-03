# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:54:30 2024

@author: WRI India, Femke, Cormac, Arpan

=========================================
ftt_s_fuel_consumption.py
=========================================
Steel production module.
####################################

This is the fuel consumption module of the FTT-Steel model which helps is calibrating the fuel
consumption in the steel sector in the historical period and is also called for the simulation
of FTT-Steel, projecting the fuel consumption in the future years. 

The outputs of this module include changes in final fuel consumption for the full steel sector.

Parameter
---------
- ftt_s_fuel_consumption(bstc, sewg, smed, nrti, nstti, c5ti, njti)
- bstc : dictionary
    steel cost matrix with three dimensions (regions, pathways/technologies, cost components/time)
- sewg : dictionary
    steel production by technology with three dimensions (regions, pathways/technologies, cost components/time)
- smed : dictionary
    steel energy density with three dimensions (regions, pathways/technologies, cost components/time)
- nrti : int
    length of RTI (region classification)
- nstti : int
    length of STTI (technology/pathways)
-c5ti : tuple
    steel cost components and there index
- njti : int
    length of JTI (Energy carrier classification)

Returns
-------
- sjef : np.array 3D
    Total consumption of materials
- sjco : np.array 3D
    Total consumption of materials for E3ME emission calculation


"""

# Third party imports
import numpy as np

def ftt_s_fuel_consumption(bstc, sewg, smed, nrti, nstti, c5ti, njti):
    sjef = np.zeros((nrti , njti, 1))  ## changed
    sjco = np.zeros_like((sjef))

    
    for r in range(nrti):
        for i in range(nstti):
        # Calculate fuel consumption

            sjef[r, 0, 0] += bstc[r, i,c5ti["Hard Coal"]] * sewg[r, i, 0]* 1000 * smed[0,11,0] * 1/41868
            sjef[r, 1, 0] += bstc[r,i,c5ti["Other Coal"]] * sewg[r, i, 0]* 1000 * smed[0,12,0] * 1/41868
            sjef[r, 6, 0] += bstc[r,i,c5ti["Natural Gas"]] * sewg[r, i, 0]* 1000 * smed[0,13,0] * 1/41868
            sjef[r, 7, 0] += bstc[r,i,c5ti["Electricity"]] * sewg[r, i, 0]* 1000 * smed[0,14,0] * 1/41868
            sjef[r, 10, 0] += ((bstc[r,i,c5ti["Biocharcoal"]] * sewg[r, i, 0]* 1000 * smed[0,18,0] * 1/41868) + (bstc[r,i,c5ti["Biogas"]] * sewg[r, i, 0]* 1000 * smed[0,19,0] * 1/41868))
            sjef[r, 11, 0] += bstc[r,i,c5ti["Hydrogen"]] * sewg[r, i, 0]* 1000 * smed[0,17,0] * 1/41868

            if (bstc[r, i, 21] == 1):

                sjco[r, 0, 0] += 0.1 * bstc[r,i,c5ti["Hard Coal"]]*sewg[r,i,0]* 1000 * smed[0,11,0]*1/41868
                sjco[r, 1, 0] += 0.1 * bstc[r,i,c5ti["Other Coal"]] * sewg[r, i, 0]* 1000 * smed[0,12,0] * 1/41868
                sjco[r, 6, 0] += 0.1 * bstc[r,i,c5ti["Natural Gas"]] * sewg[r, i, 0]* 1000 * smed[0,13,0] * 1/41868
                sjco[r, 7, 0] += bstc[r,i,c5ti["Electricity"]] * sewg[r, i, 0]* 1000 * smed[0,14,0] * 1/41868
                sjco[r, 10, 0] += -0.9 * ((bstc[r,i,c5ti["Biocharcoal"]] * sewg[r, i, 0]* 1000 * smed[0,14,0] * 1/41868) + (bstc[r,i,c5ti["Biogas"]] * sewg[r, i, 0]* 1000 * smed[0,19,0] * 1/41868))
                sjco[r, 11, 0] += bstc[r,i,c5ti["Hydrogen"]] * sewg[r, i, 0]* 1000 * smed[0,17,0] * 1/41868

            else:

                sjco[r,0,0] += bstc[r,i,c5ti["Hard Coal"]]*sewg[r,i,0]* 1000 * smed[0,11,0]*1/41868
                sjco[r,1,0] += bstc[r,i,c5ti["Other Coal"]] * sewg[r, i, 0]* 1000 * smed[0,12,0] * 1/41868
                sjco[r,6,0] += bstc[r,i,c5ti["Natural Gas"]] * sewg[r, i, 0]* 1000 * smed[0,13,0] * 1/41868
                sjco[r,7,0] += bstc[r,i,c5ti["Electricity"]] * sewg[r, i, 0]* 1000 * smed[0,14,0] * 1/41868
                sjco[r,10,0] += 0.0
                sjco[r,11,0] += bstc[r,i,c5ti["Hydrogen"]] * sewg[r, i, 0]* 1000 * smed[0,17,0] * 1/41868
    
    
    """
    Written by me
    """
    # import os

    # output_dir = 'output1'
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # # List of arrays and their corresponding names
    # arrays = [(sjef, 'sjef'), (sjco, 'sjco')]

    # # Save each array
    # for array, name in arrays:
    #     # If reshaping is needed, uncomment and adjust the next line
    #     # reshaped_array = array.reshape(-1, array.shape[-1])
    #     reshaped_array = array.reshape(array.shape[0], -1)
    #     # Construct the file name and save the array as a CSV file
    #     file_name = f"{name}.csv"
    #     file_path = os.path.join(output_dir, file_name)
        
    #     np.savetxt(file_path, reshaped_array, delimiter=",")
    #     print(f"Saved {file_path}")

    
    return sjef, sjco

