# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:26:47 2024

@author: Femke Nijsse
"""

import matplotlib.pyplot as plt


# Set global font size
plt.rcParams.update(
    {'font.size': 14,
     'legend.fontsize': 14,
     'xtick.labelsize': 14,
     'ytick.labelsize': 14,
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
