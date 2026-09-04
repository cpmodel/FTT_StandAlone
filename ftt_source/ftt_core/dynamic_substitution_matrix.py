# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 10:28:42 2026

@author: Femke
"""

import numpy as np
from numba import njit

@njit(fastmath=True, cache=True)
def compute_dynamic_subst(lifetimes, buildtimes, shares_dt, kappa, num_regions, num_techs):
    """
    Computes the substitution matrix subst[r, a, b]: the maximum rate at
    which tech a loses share to tech b, before cost/preference weighting.

    Build time (BT_b) sets how much of a's outflow a given competitor b
    can absorb relative to other competitors. kappa sets the ceiling on
    a's total outflow: under total disfavor (all competitors fully
    preferred), the outflow reduces to (kappa/LT_a) * S_a * (1 - S_a),
    maximised at S_a = 0.5, giving max(dS_a/dt) = kappa / (4 * LT_a).

    kappa = 2 reaches 1/(2*LT_a) at that maximum (even age distribution
    between old and new capital, so half of retiring capital transfers).
    kappa = 4 reaches 1/LT_a (full transfer of a's entire retiring stock
    to the new technology). kappa = 3 is a reasonable middle value; test
    sensitivity in [2, 4].
    """

    subst = np.zeros((num_regions, num_techs, num_techs))

    for r in range(num_regions):
        for a in range(num_techs):

            S_a = shares_dt[r, a, 0]
            LT_a = lifetimes[r, a]

            if not (S_a > 0.0 and LT_a > 0.0):
                continue

            # Share-weighted sum of competitor build-time rates, and total
            # competitor share, used to normalise the aggregate outflow.
            competitor_share_over_buildtime = 0.0
            competitor_share_total = 0.0

            for b in range(num_techs):
                if b == a:
                    continue
                S_b = shares_dt[r, b, 0]
                BT_b = buildtimes[r, b]
                if S_b > 0.0 and BT_b > 0.0:
                    competitor_share_over_buildtime += S_b / BT_b
                    competitor_share_total += S_b

            if competitor_share_over_buildtime <= 0.0 or competitor_share_total <= 0.0:
                continue

            kappa_effective = kappa * competitor_share_total / competitor_share_over_buildtime
            
            # Final substitution calculations
            for b in range(num_techs):
                if b == a:
                    continue
                BT_b = buildtimes[r, b]
                if BT_b > 0.0:
                    subst[r, b, a] = kappa_effective / (LT_a * BT_b)

    return subst