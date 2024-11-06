# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:26:47 2024

@author: Femke Nijsse
"""

import matplotlib.pyplot as plt


# Set global font size
plt.rcParams.update(
    {'font.size': 7,
     'figure.titlesize': 7, 
     'axes.titlesize': 7,
     'font.family': 'sans-serif',
     'font.sans-serif': ['Helvetica', 'Arial'],  # Tries Helvetica first, then Arial
     'legend.fontsize': 7,
     'xtick.labelsize': 7,
     'ytick.labelsize': 7,
     'axes.labelsize': 7,
     'legend.title_fontsize': 7,  # Legend title font size to match
     'figure.dpi': 300})


# Mappings
PRICE_NAMES = {
    "FTT:P": "MECW battery only",
    "FTT:Tr": "TEWC",
    "FTT:H": "HEWC",
    "FTT:Fr": "ZTLC"
}

TECH_VARIABLE = {
    "FTT:P": 18,
    "FTT:Tr": 19,
    "FTT:H": 10,
    "FTT:Fr": 12
}

OPERATION_COST_NAME = {
    "FTT:P": "MLCO"
}

REPL_DICT = {"FTT:P": "Power",
             "FTT:H": "Heat",
             "FTT:Tr": "Cars",
             "FTT:Fr": "Trucks"}

REPL_DICT2 = {"FTT:P": "Power: \n new clean tech vs existing fossil tech",
             "FTT:H": "Heat",
             "FTT:Tr": "Cars",
             "FTT:Fr": "Large trucks"}
