# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:14:12 2023

Output manipulation - to be used after model run

@author: ib400
"""

s0 = output_all['S0']
mcocx = s0['MCOCX']
mcoc = s0['MCOC']
mcocx_be = mcocx[0,:,0,:]
mcoc_be = mcoc[0,:,0,:]

mcocx_us = mcocx[37,:,0,:]
mcoc_us = mcoc[37,:,0,:]

mewg = s0['MEWG']
mewgx = s0['MEWGX']

mewg_us = mewg[37,:,0,:]
mewgx_us = mewgx[37,:,0,:]
repp = s0['REPP'][:,0,0,:]
