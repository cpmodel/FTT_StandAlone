"""
Module contains functions to generate plots for C40 cities e-truck report.

"""

# Standard library imports
import os
import sys
import matplotlib.pyplot as plt

# local imports
from plot_code.plot_stacked_area import plot_zewk_hdt_stacked
from plot_code.plot_subsidy import plot_ztvt_timeseries
from plot_code.plot_tco_parity import plot_mandate_tco
from plot_code.plot_current_traj import plot_costs, plot_shares

# create output directory if it doesn't exist
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# change dir to parent
os.chdir('..')

plt.rcParams['font.family'] = 'outfit'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


plot_options = ["mandate_tco", "subsidy", "policy_effect", "costs", "shares"]
# Get cmd line arguments
arguments = sys.argv[1:]
if not arguments:
    print(f"No plots selected. Rendering all available plots...")
    selected_plots = plot_options
else:
    selected_plots = [arg for arg in arguments if arg in plot_options]

if "mandate_tco" in selected_plots:
    plot_mandate_tco(
    regions={41:"China", 3:"Germany", 42:"India",  8:"Italy", 5:"Spain", 15:"UK"},
    baseline_name={'S0': 'Baseline'},
    scenario_names={'city_mandates_2030': '100pct sales by 2030',
                    'city_mandates_2035': '100pct sales by 2035',
                    'city_mandates_2040': '100pct sales by 2040'},
    pickle_name='Results_mandates'
    )
    print("Mandate TCO plot created.")


if "subsidy" in selected_plots:
    plot_ztvt_timeseries(
        regions={82:"Berlin", 115:"Delhi", 87:"Milan", 110:"London", 95:"Rotterdam"},
        scenario_name={'tco_parity': 'TCO Parity'},
        pickle_name='Results_stacked_figure',
        tech_indices=(32, 33)
    )
    print("Subsidy plot created.")

if "policy_effect" in selected_plots:
    plot_zewk_hdt_stacked(
        regions={72:"Beijing", 82:"Berlin", 115:"Delhi", 110:"London", 98:"Madrid", 87:"Milan"},
        scenarios={'S0': 'Baseline',
                   "carbon_tax": "Carbon tax",
                   "tco_parity": "TCO feebate",
                   'city_mandates_2040': 'City mandates'},
        pickle_name='Results_stacked_figure',
    )
    print("Policy effectiveness plot created.")

if "costs" in selected_plots:
    plot_costs(
        regions={72:"Beijing", 82:"Berlin", 115:"Delhi", 110:"London", 98:"Madrid", 87:"Milan"},
        scenario_name_map={'S0': 'Baseline'},
        scenario_type_map={'S0': 'main', 'S1': 'positive', 'S2': 'negative'},
        pickle_name='Results_sensitivities',
    )
    print("Cost trajectory plot created.")
    
if "shares" in selected_plots:  
    plot_shares(
        regions={72:"Beijing", 82:"Berlin", 115:"Delhi", 110:"London", 98:"Madrid", 87:"Milan"},
        scenario_name_map={'S0': 'Baseline'},
        scenario_type_map={'S0': 'main', 'S1': 'positive', 'S2': 'negative'},
        pickle_name='Results_sensitivities',
    )
    print("Share trajectory plot created.")