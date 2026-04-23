# -*- coding: utf-8 -*-
"""
Centralized emissions regulation functions for FTT sectors.

Single entry point: `implement_emissions_regulation`. Mirrors the flow of
`ftt_mandate.py`, sectors call it each sub-timestep of the annual loop
with their current sales/capacity arrays plus a per-region info array, and
this module handles segment/class grouping, eligibility, per-region
timelines, and baseline caching. The target depends only on the integer
`year` and the cached baseline, so repeated within-year calls share the
 same target and are deterministic.

Regulation info (per region):
    [start_year, end_year, max_reduction]
    - max_reduction = 1.0  → target reaches 0 at end_year
    - max_reduction = 0.5  → target reaches 50% of baseline at end_year
    - max_reduction <= 0   → regulation off for that region

Behaviour:
    - Baseline is the fleet-average CO2 at the region's start_year, computed
      live and cached across years in this module.
    - Target declines linearly from baseline to baseline * (1 - max_reduction)
      over [start_year, end_year]. After end_year the target stays at that
      floor (regulation persists, does not expire).
    - Sales shift from high-emitting techs to eligible low-emitting techs
      proportionally to their existing share of the eligible pool.

Functions:
    get_fleet_emissions: Weighted average fleet CO2
    implement_emissions_regulation: Main entry point
    reset_baseline_cache: Clear cached baselines (for tests / fresh runs)

@author: Amir Akther
"""

import numpy as np


# --- Sector configuration ---------------------------------------------------

# Passenger transport segments. Indices are into the flat 31-tech VTTI array.
# Eligible receivers: Electric, PHEV only. The policy is labelled "EV
# regulation" so donations should concentrate on EVs rather than siphoning
# into mature Hybrid (which holds non-trivial share in markets like China
# and would otherwise capture most of the proportional redistribution).
# Everything else — Petrol, Adv Petrol, Diesel, Adv Diesel, CNG, Hybrid —
# is a potential donor once target dips below its emissions; Hydrogen
# stays neutral because its emissions are 0.
_TRANSPORT_SEGMENTS = {
    'econ': {'indices': [0, 3, 6, 9, 12, 15, 18, 21, 24],
             'eligible': [18, 21]},
    'mid':  {'indices': [1, 4, 7, 10, 13, 16, 19, 22, 25],
             'eligible': [19, 22]},
    'lux':  {'indices': [2, 5, 8, 11, 14, 17, 20, 23, 26],
             'eligible': [20, 23]},
}

# Freight: 5 vehicle classes (TWV, LCV, MDT, HDT, Bus), 9 techs each,
# interleaved in the 45-tech FTTI array. Per-class tech order:
# 0 Petrol, 1 Adv petrol, 2 Diesel, 3 Adv diesel, 4 CNG/LPG,
# 5 PHEV, 6 BEV, 7 Bioethanol, 8 FCEV.
# Eligible receivers: Adv petrol, Adv diesel, PHEV, BEV, Bioethanol.
# Excluded: CNG/LPG, FCEV (mirrors Transport's gas + hydrogen exclusion).
_FREIGHT_N_CLASSES = 5
_FREIGHT_ELIGIBLE = [1, 3, 5, 6, 7]

_SECTOR_CONFIG = {
    'transport': {'layout': 'grouped', 'segments': _TRANSPORT_SEGMENTS},
    'freight':   {'layout': 'interleaved',
                  'n_classes': _FREIGHT_N_CLASSES,
                  'eligible': _FREIGHT_ELIGIBLE},
}

# Module-level baseline cache: sector -> (n_regions, n_classes) array.
# NaN entries = "not yet computed". Populated the first time regulation is
# active for that region. Persists across years within a simulation run.
_baseline_cache = {}


def reset_baseline_cache():
    """Clear cached baselines. Call between independent simulation runs."""
    _baseline_cache.clear()


# --- Core numerics ----------------------------------------------------------

def get_fleet_emissions(sales, emissions_intensity):
    """Weighted average fleet emissions. Returns 0 when total sales are 0."""
    total_sales = np.sum(sales)
    if total_sales == 0:
        return 0.0
    return np.sum(sales * emissions_intensity) / total_sales


def _redistribute_proportionally(sales, emissions_intensity, target, eligible_mask):
    """
    Shift sales from non-eligible above-target techs to eligible techs.

    Eligible techs are always treated as protected destinations: they
    never lose sales to the policy, regardless of their emissions rank.
    This matters when the emissions data uses well-to-wheel accounting
    (e.g. Transport, where EVs have non-zero grid emissions and PHEV can
    sit below EV in the ranking). Without this rule, a declining target
    would eventually push sales from EVs to PHEVs — the opposite of what
    EV-regulation policy intends.

    Donors: non-eligible techs with emissions above the target.
    Receivers: eligible techs, weighted by current share.
    Non-eligible below-target techs (e.g. Hydrogen in Transport) are
    left untouched.
    """
    sales_out = np.copy(sales)

    if np.sum(sales_out) == 0:
        return sales_out
    if get_fleet_emissions(sales_out, emissions_intensity) <= target:
        return sales_out

    donor_mask = (emissions_intensity > target) & ~eligible_mask
    receiver_mask = eligible_mask

    if not np.any(donor_mask) or not np.any(receiver_mask):
        return sales_out

    donor_indices = np.where(donor_mask)[0]
    receiver_indices = np.where(receiver_mask)[0]

    for _ in range(100):
        current = get_fleet_emissions(sales_out, emissions_intensity)
        if current <= target:
            break

        shift_fraction = min(0.1, (current - target) / current)
        donor_sales = np.sum(sales_out[donor_mask])
        if donor_sales == 0:
            break
        shift_amount = donor_sales * shift_fraction

        for i in donor_indices:
            if sales_out[i] > 0:
                sales_out[i] -= (sales_out[i] / donor_sales) * shift_amount

        receiver_sales = np.sum(sales_out[receiver_mask])
        if receiver_sales > 0:
            for i in receiver_indices:
                sales_out[i] += (sales_out[i] / receiver_sales) * shift_amount
        else:
            per_tech = shift_amount / len(receiver_indices)
            for i in receiver_indices:
                sales_out[i] += per_tech

    return sales_out


def _target_emissions(baseline, year, start_year, end_year, max_reduction):
    """
    Declining target: baseline at start_year, baseline*(1-max_reduction)
    at end_year, held at that floor thereafter.
    """
    total_years = end_year - start_year
    if total_years <= 0:
        return baseline
    progress = min(max(year - start_year, 0), total_years) / total_years
    return baseline * (1 - max_reduction * progress)


def _regulate_one_class(cap, cum_sales_in, sales_in, year,
                        emissions_intensity, eligible_indices,
                        start_years, end_years, max_reductions, baseline):
    """
    Apply regulation to one already-sliced class/segment.
    Returns new (cum, sales, cap) triple for the slice.
    """
    sales_after = np.copy(sales_in)
    eligible_mask = np.zeros(sales_in.shape[1], dtype=bool)
    eligible_mask[eligible_indices] = True

    for r in range(sales_in.shape[0]):
        start_r = start_years[r]
        end_r = end_years[r]
        max_red_r = max_reductions[r]

        if max_red_r <= 0 or year < start_r or end_r <= start_r:
            continue

        em_r = emissions_intensity[r, :]
        sales_r = sales_in[r, :, 0]

        if np.isnan(baseline[r]):
            baseline[r] = get_fleet_emissions(sales_r, em_r)

        target = _target_emissions(baseline[r], year, start_r, end_r, max_red_r)

        sales_after[r, :, 0] = _redistribute_proportionally(
            sales_r, em_r, target, eligible_mask
        )

    sales_diff = sales_after - sales_in
    cap_after = cap + sales_diff
    cap_after[:, :, 0] = np.maximum(cap_after[:, :, 0], 0)
    cum_sales_after = np.copy(cum_sales_in)
    cum_sales_after[:, :, 0] += sales_diff[:, :, 0]
    return cum_sales_after, sales_after, cap_after


# --- Public entry point -----------------------------------------------------

def implement_emissions_regulation(cap, cum_sales_in, sales_in, year,
                                   emissions_intensity, regulation_info,
                                   sector):
    """
    Apply declining fleet-average CO2 regulation to a sector.

    Parameters
    ----------
    cap : ndarray
        Full capacity array (regions x techs x 1).
    cum_sales_in, sales_in : ndarray
        Full cumulative / current-period sales (regions x techs x 1).
    year : int
        Simulation year.
    emissions_intensity : ndarray
        CO2 per tech (regions x techs) in gCO2/km.
    regulation_info : ndarray
        Per-region policy info, shape (regions, 3, 1) or (regions, 3):
        columns [start_year, end_year, max_reduction]. max_reduction <= 0
        disables regulation for that region.
    sector : str
        'transport' or 'freight'. Selects the internal grouping and
        eligibility configuration.

    Returns
    -------
    (cum_sales_after, sales_after, cap_after)
    """
    if sector not in _SECTOR_CONFIG:
        raise ValueError(f"Unknown sector '{sector}'. "
                         f"Expected one of {list(_SECTOR_CONFIG)}.")

    # Accept (regions, 3), (regions, 3, 1), or (regions, 3, 1, 1) — the loader
    # produces 4D with trailing singleton dims. Collapse to strict (regions, 3).
    info = np.asarray(regulation_info, dtype=float).reshape(
        np.asarray(regulation_info).shape[0], -1
    )[:, :3]
    start_years = info[:, 0]
    end_years = info[:, 1]
    max_reductions = info[:, 2]

    # Early exit if no region has active regulation this year
    active = (max_reductions > 0) & (year >= start_years) & (end_years > start_years)
    if not np.any(active):
        return cum_sales_in, sales_in, cap

    config = _SECTOR_CONFIG[sector]
    n_regions = sales_in.shape[0]

    cum_out = np.copy(cum_sales_in)
    sales_out = np.copy(sales_in)
    cap_out = np.copy(cap)

    if config['layout'] == 'grouped':
        segments = config['segments']
        if (sector not in _baseline_cache
                or _baseline_cache[sector].shape != (n_regions, len(segments))):
            _baseline_cache[sector] = np.full((n_regions, len(segments)), np.nan)
        cache = _baseline_cache[sector]

        for seg_idx, seg_cfg in enumerate(segments.values()):
            indices = seg_cfg['indices']
            eligible_local = [indices.index(e) for e in seg_cfg['eligible']]

            cum_s, sales_s, cap_s = _regulate_one_class(
                cap_out[:, indices, :],
                cum_out[:, indices, :],
                sales_out[:, indices, :],
                year,
                emissions_intensity[:, indices],
                eligible_local,
                start_years, end_years, max_reductions,
                cache[:, seg_idx],
            )
            cum_out[:, indices, :] = cum_s
            sales_out[:, indices, :] = sales_s
            cap_out[:, indices, :] = cap_s

    elif config['layout'] == 'interleaved':
        n_classes = config['n_classes']
        eligible_local = config['eligible']
        if (sector not in _baseline_cache
                or _baseline_cache[sector].shape != (n_regions, n_classes)):
            _baseline_cache[sector] = np.full((n_regions, n_classes), np.nan)
        cache = _baseline_cache[sector]

        for v_class in range(n_classes):
            idx = slice(v_class, None, n_classes)

            cum_s, sales_s, cap_s = _regulate_one_class(
                cap_out[:, idx, :],
                cum_out[:, idx, :],
                sales_out[:, idx, :],
                year,
                emissions_intensity[:, idx],
                eligible_local,
                start_years, end_years, max_reductions,
                cache[:, v_class],
            )
            cum_out[:, idx, :] = cum_s
            sales_out[:, idx, :] = sales_s
            cap_out[:, idx, :] = cap_s

    return cum_out, sales_out, cap_out
