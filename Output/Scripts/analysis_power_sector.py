# -*- coding: utf-8 -*-
"""
Description: Analyse output for the paper  "Is a solar future inevitable?"
    
@author: Femke
"""
import os
from celib import DB1   # The CE library
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pylab as pylab
from pathlib import Path
import matplotlib as mpl
import math
from matplotlib.colors import ListedColormap

from E3MEpackage import country_to_E3ME_region70, E3ME_regions_names70
from preprocessing import get_df

# This helps with the bug that seaborn seems to override mpl rcParams
import importlib
importlib.reload(mpl); importlib.reload(plt); importlib.reload(sns)


params = {"font.family": "Arial",
          'font.size': 7}
           #'legend.fontsize': 12,
#          # 'figure.figsize': (7, 5),
#          'axes.labelsize': 15,
#          'axes.titlesize':16,
#          'xtick.labelsize':12,
#          'ytick.labelsize':12,
#          'text.usetex': False,
#          "svg.fonttype": 'none'}
pylab.rcParams.update(params)
sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 1})
start_year = 2015

figsize_inches_one_column = 3.46457
figsize_inches_two_column = 7.08661
#%%
if __name__ == '__main__':
    
    do_plot = True
    figures_directory = "C://Users\Work profile\OneDrive - University of Exeter\Documents\Work\Papers\Solar inevitable\Figures\\Da_63_"
    source_data_directory = "C://Users\Work profile\OneDrive - University of Exeter\Documents\Work\Papers\Solar inevitable\Source Data//"

    # This script can be saved in post-processing which is on the same level as Master
    dirp_graph = os.path.dirname(os.path.realpath(__file__))
    dirp = Path(dirp_graph).parents[1]
    dir_db = os.path.join(dirp, 'databank')
    dirp_out = os.path.join(dirp, 'Output')
    
    udbp = os.path.join(dir_db, 'U.db1')
    with DB1(udbp) as db1:
        rti = db1['RSHORTTI']   # Two-letter region codes
        t2ti = db1['T2TI']      # Names of the power sector technologies
        lbti = db1['LBTI']      # Name of the loadbands for dispatchable technologies
        jti = db1['JTI']        # Name of fuels used
        yti = db1['YTI']        # Name of sectors in EU
        yrti = db1["YRTI"]      # Name of sectors outside EU (less disaggregated)
    lbti = [lb[2:] for lb in lbti]

    
    scenarios = {
                'Baseline': 'Dan_ba.mre',
                'Baseline_new': 'Dan_ba60b.mre',
                #'Baseline_new': 'Dan_ba63.mre',
                }

    colour_list = ['cyan', 'lightsalmon', 'black', 'darkgrey', 'purple',
                   'violet', 'forestgreen', 'lime', 'darkblue', 'deepskyblue',
                   'gold', 'orange', 'brown', 'lightcoral']
    ce_cols = ['#0B1F2C', '#909090', '#C5446E', '#49C9C5', '#AAB71D', '#009FE3']
    colour_map_lbs = dict(zip(lbti, ce_cols))

    regs_to_print = {
                     'EU-27': [x for x in range(33) if (x < 27 or x == 32) and x != 14],
                     'USA': [33],
                     'India': [41],
                     'China': [40],
                     #'Japan': [34],
                     #'Brazil': [43],
                     'Russia': [38],
                     'Africa': [x for x in range(58, 70) if x not in [60, 61]],
                     'Global': list(range(len(rti)))
                     }
  
    # EU countries
    # regs_to_print2 = {"Germany": [2],
    #                  "France": [5],
    #                  "United Kingdom": [14],
    #                  "Spain": [4],
    #                  "Italy": [7]}
    my_map = E3ME_regions_names70()
    
    # Comment this if you want to process the eight countries above only
    # regs_to_print = dict((v, [k-1]) for k, v in my_map.items())

    scenarios_to_print = [#"Scenario A",
                         "Baseline_new"]

    #%%
    # Get the data from running the preprocessing script
    df, df_shares, df_loadband, df_capacity, df_generation = get_df(scenarios, scenarios_to_print, start_year, dirp_out, regs_to_print, 
                           print_temperature=True)
    
    
    colours = {'Nuclear': 'sienna', "Coal": 'grey', "Oil": 'darkgrey', "Gas": 'lightgrey',
               "Bioenergy":'C2', 'Other': 'C8','Hydro': 'paleturquoise', 'Hydro':'paleturquoise',
               'Onshore wind': 'cornflowerblue', "Offshore wind": 'royalblue',
               'Solar PV': 'salmon', "Solar": 'salmon', 'CSP': 'gold'}
    
    
    if len(regs_to_print) < 50:    
        #%% =======================================================================
        #       Plot share of generation in electricity mix
        # =========================================================================
        
        df_shares_global = df_shares[df_shares['Region']=='Global']
        df_generation_global = df_generation[df_generation['Region']=='Global']
        df_shares_global.to_csv(source_data_directory + "Figure1.csv", index=False)

        
        # Various statements for in the text
        print(f"Maximum share onshore occurs in {df_shares_global['Year'][df_shares_global['Onshore wind'].idxmax()]}")
        print(f"Maximum share offshore occurs in {df_shares_global['Year'][df_shares_global['Offshore wind'].idxmax()]}")
        print(f'{sum(np.array(df_shares_global.query("Year==2020"))[0, 3:]>10)} technologies provide more than 10% of electricity in 2020')
        print(f'{sum(np.array(df_shares_global.query("Year==2050"))[0, 3:]>10)} technologies provide more than 10% of electricity in 2050')
        print(f"Maximum generation onshore occurs in {df_generation_global['Year'][df_generation_global['Generation onshore wind'].idxmax()]}")
        print(f"Maximum generation offshore occurs in {df_generation_global['Year'][df_generation_global['Generation offshore wind'].idxmax()]}")
        df_shares.query('Year==2020 and Region=="Global"')
        dfs_2020g = df_shares.query('Year==2020 and Region=="Global"')
        dfs_2050g = df_shares.query('Year==2050 and Region=="Global"')
        print(f"In 2020, fossil fuels provide {float(dfs_2020g['Gas']+dfs_2020g['Oil']+dfs_2020g['Coal']):.1f}% of electricity generation")
        print(f"In 2050, fossil fuels provide {float(dfs_2050g['Gas']+dfs_2050g['Oil']+dfs_2050g['Coal']):.1f}% of electricity generation")
        print(f"In 2050, solar PV&CSP provide {float(dfs_2050g['Solar PV'] + dfs_2050g['CSP']):.1f}%")
        
        
        # Figure 1: Plot global shares of generation      
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(figsize_inches_one_column, figsize_inches_one_column*0.8))
        df_shares_global[df_shares_global['Scenario']=="Baseline_new"].plot.area(x="Year", color=colours, ax=ax, fontsize=7)
        lgd = ax.legend(loc=(-0.07, -0.35), ncol=4, fontsize=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis='y', which='major', pad=-10)
        ax.tick_params(axis='y', which=u'both',length=0)
        ax.set_yticks([25, 50, 75, 100], [25, 50, 75, 100], minor=False, fontsize=7) 
        years_labels = np.arange(math.ceil(start_year/10)*10, 2061, 10)
        ax.set_xticks(years_labels, years_labels, minor=False, fontsize=7)  
        ax.set_ylim(0, 105)
        
        fig = ax.get_figure()
        fig.savefig(figures_directory + "Figure1.svg", bbox_extra_artists=(lgd,), bbox_inches='tight')
        
    
        # Figure SI 1: Showing regional shares of generation
        fig, axs = plt.subplots(2, 4, figsize=(12,6), sharey=True)
        axs = axs.ravel()    # Making it easier to loop over
        for i, region in enumerate(regs_to_print.keys()):
            if region != "Global":
                df_shares_region = df_shares[df_shares['Region']==region]
                ax = axs[i]
                df_shares_region[df_shares_region['Scenario']=="Baseline_new"].plot.area(x="Year", color=colours, ax=ax, legend=False)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.tick_params(axis='y', which='major', pad=-2)
                ax.tick_params(axis='y', which=u'both',length=0)
                ax.set_yticks([25, 50, 75, 100], [25, 50, 75, 100], minor=False, fontsize=12)   
                ax.set_xticks([2020, 2040, 2060], [2020, 2040, 2060], minor=False, fontsize=12)  
                ax.set_ylim(0, 105)
                ax.set_title(region, pad=-2, y=1.000001) # y=1.0.. because bug (https://github.com/matplotlib/matplotlib/issues/16805/)
                ax.set_xlabel("")
        lgd = ax.legend(loc=(-2.7, -0.55), ncol=4, fontsize=12)
        fig.subplots_adjust(wspace=0.1)
        fig.savefig(figures_directory + "FigureSI1.svg", bbox_extra_artists=(lgd,), bbox_inches='tight')

     
               
        
       
        #%% Plotting CO2 emissions total (so including non-power)
    
        g = sns.relplot(data=df, x="Year", y=r'CO$_2$ emissions w.r.t. 2010', hue="Region", kind='line',
                     col='Scenario', col_wrap=1, legend=False)
        (g.map(plt.axhline, y=0, color=".7", alpha=0, dashes=(2, 1), zorder=0)
        .set_axis_labels("Year", "Percentage")
        .set_titles(r'CO$_2$ emissions w.r.t. 2010')
        .tight_layout(h_pad=5, w_pad=1))
        #g.axes[0].annotate("Baseline", ((0, 1)), xycoords='figure fraction')
        
        #%% Plotting which countries have driven price declines
        
        technology = "Solar"
        df_capacity_solar = df_capacity[df_capacity["Technology"] == technology]
        df_capacity_solar = df_capacity_solar[df_capacity_solar["Year"] < 2035]
        df_capacity_solar = df_capacity_solar[df_capacity_solar["Year"] %5 == 0]

        df_capacity_solar = df_capacity_solar.set_index("Year")
        ax = df_capacity_solar.plot(kind='bar', stacked=True)
        ax.xaxis.label.set_visible(False)
        ax.legend(loc=(-0.0, -0.30), ncol=4, fontsize=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis='y', which='major', pad=-10)
        ax.tick_params(axis='y', which=u'both',length=0)
        ax.set_yticks([25, 50, 75, 100],[25, 50, 75, 100], minor=False, fontsize=7)   
        #ax.set_xticks([2020, 2030, 2040, 2050, 2060], minor=False, fontsize=7)  
        ax.set_ylim(0, 105)
        ax.set_title(f"{technology} share of installed capacity")
        
        
        #%%
        # Figure 6: Plotting investments
        my_colours = sns.color_palette("muted").as_hex()
        
       
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(figsize_inches_two_column, figsize_inches_two_column/2.7))
        df_reg = df[df["Region"]=="Global"]
        axes[0].plot(df_reg["Year"], df_reg["Investment"], c="black", lw=2, label="Global")
        axes[1].plot(df_reg["Year"], df_reg["Investment wrt 2019 benchmark"], c="black", lw=2, label="Global")
        df_reg[["Year", "Investment", "Investment wrt 2019 benchmark"]].to_csv(source_data_directory + "Figure6_Global.csv", index=False)

        for i, region in enumerate(regs_to_print.keys()):
            df_reg = df[df["Region"]==region]
            if region != "Global":
                axes[0].plot(df_reg["Year"], df_reg["Investment"], label=region, lw=1, c=my_colours[i])
                axes[1].plot(df_reg["Year"], df_reg["Investment wrt 2019 benchmark"], label=region, lw=1, c=my_colours[i])
                df_reg[["Year", "Investment", "Investment wrt 2019 benchmark"]].to_csv(source_data_directory + f"Figure6_{region}.csv", index=False)

            
        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, c='grey', alpha=0.2)
            ax.set_xticks([2020, 2030, 2040, 2050, 2060], [2020, 2030, 2040, 2050, 2060],  minor=False, fontsize=7)   
            ax.set_xlim(start_year, 2060)
        axes[0].set_ylim(0, 4)
        axes[1].set_ylim(0, 10)
        axes[0].set_title("Investment as percentage of GDP", fontsize=7)
        axes[1].set_title("Investment wrt 2019", fontsize=7)
        axes[0].legend(prop={'size': 7})
        
        axes[0].annotate("a", xy=(0.05, 0.93), xycoords='axes fraction', fontweight='bold', fontsize=7)
        axes[1].annotate("b", xy=(0.05, 0.93), xycoords='axes fraction', fontweight='bold', fontsize=7)
        
        plt.show()
        
        
       #%%  Figure 2: share of renewables and of solar PV        
       
        g = sns.relplot(data=df[df['Region']!='Global'], x="Year", y=r'Electricity demand', hue="Region", kind='line',
                     col='Scenario', col_wrap=1)
        (g.map(plt.axhline, y=0, color=".7", alpha=0, dashes=(2, 1), zorder=0)
        .set_axis_labels("Year", "Thousand toe")
        .set_titles(r'Electricity demand')
        .tight_layout(h_pad=5, w_pad=1))
        g.axes[0].set_xticks([2020, 2030, 2040, 2050, 2060], [2020, 2030, 2040, 2050, 2060], minor=False)   
    
        
        
        # Shares renewables and solar PV
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(figsize_inches_two_column, figsize_inches_two_column/2.7))
        df_reg = df[df["Region"]=="Global"]
        axes[0].plot(df_reg["Year"], df_reg["Share renewables power"], c="black", lw=1.5, label="Global")
        axes[1].plot(df_reg["Year"], df_reg["Share solar"], c="black", lw=1.5, label="Global")
        df_reg[["Year", "Share renewables power", "Share solar"]].to_csv(source_data_directory + "Figure2_Global.csv", index=False)

        for i, region in enumerate(regs_to_print.keys()):
            df_reg = df[df["Region"]==region]
            if region != "Global":
                axes[0].plot(df_reg["Year"], df_reg["Share renewables power"], label=region, lw=1, c=my_colours[i])
                axes[1].plot(df_reg["Year"], df_reg["Share solar"], label=region, lw=1, c=my_colours[i])
                df_reg[["Year", "Share renewables power", "Share solar"]].to_csv(source_data_directory + f"Figure2_{region}.csv", index=False)


        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, c='grey', alpha=0.2)
            #ax.xaxis.label.set_color('dimgrey')
            #ax.tick_params(axis='both', colors='dimgrey')
            ax.set_xticks([2020, 2030, 2040, 2050, 2060], [2020, 2030, 2040, 2050, 2060], minor=False, fontsize=7)   
            ax.set_xlim(start_year, 2060)
        #axes[0].set_ylim(0, 14)
        #axes[1].set_ylim(0, 10)
        axes[0].set_title("Share renewables", fontsize=7)
        axes[1].set_title("Share solar PV", fontsize=7)
        axes[0].legend(prop={'size': 7})
        
        axes[0].annotate("a", xy=(0.05, 0.93), xycoords='axes fraction', fontweight='bold', fontsize=7)
        axes[1].annotate("b", xy=(0.05, 0.93), xycoords='axes fraction', fontweight='bold', fontsize=7)
       
        plt.show()
        
        print("Confirm this is executed")        
        
        
        #%% Plotting the LCOE of solar vs coal
        
        samplemelt = pd.melt(df, id_vars=['Year', 'Scenario', 'Region'],
                             value_vars=['LCOE solar'],
                             var_name='variable', value_name="costs")
        gm = sns.relplot(data=samplemelt, row="Scenario", y="costs", x="Year", 
                     col="variable", hue="Region", facet_kws={"sharex":"col"}, kind='line')
        
        # Changing the titles of the graphs, as the default is very ugly / overlapping
        (gm.map(plt.axhline, y=0, color=".7", alpha=0, dashes=(2, 1), zorder=0)
        .set_axis_labels("Year", "costs")
        .set_titles("{col_name}")
        .tight_layout(h_pad=5, w_pad=1))
        plt.setp(gm._legend.get_title(), fontsize=7)   
        
        
        plt.show()
        
        #%% Break-even point with gas
        
        def plot_break_even(region, scenario, fossil, bare=False, ax=None):
            inds = (df['Region']==region) & (df['Scenario']==scenario)
            if fossil == "Gas":
                fos_colour = 'lightgrey'
            if fossil == "Coal":
                fos_colour = 'grey'
            if bare:
                LCOE_solar = 'LCOE bare solar'
                if fossil == 'Gas':
                    LCOE_fossil = "LCOE bare gas"
                else:
                    LCOE_fossil = 'LCOE bare coal'
            else:
                LCOE_solar = 'LCOE solar'
                if fossil == 'Gas':
                    LCOE_fossil = "LCOE gas"
                else:
                    LCOE_fossil = 'LCOE coal'
            if ax==None:
                fig, ax = plt.subplots()
            if scenario == 'Baseline':
                colours = ['C1', 'grey']
                labs = ['Solar-old', fossil]
            elif scenario ==  'Baseline_new':
                colours = ['salmon', 'grey']
                labs = ['Solar', fossil]
            elif scenario == "Baseline-high":
                colours = ["purple", 'teal']
                labs = ["Solar-high", fossil + "-high"]
            elif scenario in ["Cheap-storage", 'Cheap-storage2', 'Cheap-storage3']:
                colours = ["darkred", "darkgrey"]
                labs = ["Solar-high-ch", fossil + '-high-ch']
            ax.plot(df['Year'][inds], df[LCOE_solar][inds], label=labs[0], c=colours[0], lw=3)
            ax.plot(df['Year'][inds], df[LCOE_fossil][inds], label=labs[1], c=fos_colour, lw=3)
            ax.set_xticks([2020, 2030, 2040, 2050, 2060], [2020, 2030, 2040, 2050, 2060], minor=False)   
            ax.legend()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylabel("LCOE (2013$/MWh)")
            return ax
        
        
        def plot_break_even_per_country(scenario, fossil, ax):
            inds = (df['Scenario'] == scenario) 
            dfl = df[inds]
            break_even_dictionary = {}
    
    
            if fossil == 'Gas':
                LCOE_fossil = "LCOE gas"
                fos_colour = 'lightgrey'
            else:
                LCOE_fossil = 'LCOE coal'
                fos_colour = 'grey'
            for n, reg in enumerate(regs_to_print):
                break_even_point_found = False
                second_break_found = False
                ind_initial = (dfl["Year"]==2020) & (dfl["Region"]==reg)
                # Is solar more expensive to start with?
                if (dfl['LCOE solar'][ind_initial] > dfl[LCOE_fossil][ind_initial]).all():
                    # Does solar become cheaper than fossil?
                    ind_final = (dfl["Year"]==2060) & (dfl["Region"]==reg)
                    
                    for y in np.arange(2020, 2061):
                        ind_y = (dfl["Year"]==y) & (dfl["Region"]==reg)
                        if (dfl['LCOE solar'][ind_y] < dfl[LCOE_fossil][ind_y]).all():
                            break_even_dictionary[reg] = y
                            #print(sum(ind_y))
                            break_even_point_found = True
                            break
                   
                    if (dfl['LCOE solar'][ind_final] > dfl[LCOE_fossil][ind_final]).all():
                        if break_even_point_found:
                        # If solar are becoming more expensive again, but a temperorary break was found
                            for y in np.arange(break_even_dictionary[reg], 2061):
                                ind_y = (dfl["Year"]==y) & (dfl["Region"]==reg)
                                if (dfl['LCOE solar'][ind_y] > dfl[LCOE_fossil][ind_y]).all():
                                    break_even_dictionary[reg+"2"] = y
                                    #print(sum(ind_y))
                                    second_break_found = True
                                    break
                            print("Need to implement a second switch")
                    
                        
                else:
                    break_even_dictionary[reg] = 2020 # Setting it to 2020 if <2020
                
                # The actual plotting
                try:
                    ax.hlines(y=0.2-n, xmin=break_even_dictionary[reg], xmax=2060, linewidth=3.5, color='salmon')
                    ax.hlines(y=0.2-n, xmin=2020, xmax=break_even_dictionary[reg], linewidth=3.5, color=fos_colour)
                    if second_break_found:
                        ax.hlines(y=0.2-n, xmin=break_even_dictionary[reg+"2"], xmax=2060, linewidth=3.5, color=fos_colour)
                except KeyError:
                    print(break_even_dictionary)
                    print(reg)
                    print("LCOE solar:")
                    print(dfl['LCOE solar'])
                    print("LCOE coal")
                    print(dfl[LCOE_fossil])
                    raise
    
                ax.set_xlim(2018, 2060)
                ax.annotate(reg, xy=(2020, -n), xytext=(2013, -n), fontsize=15)
            print("Break even year for " + fossil)    
            print(break_even_dictionary)
            ax.axis('off')
            return ax
                            
        
        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 2]})
    
        #ax1 = plot_break_even("Global", "Baseline", fossil='Gas')
        plot_break_even("Global", "Baseline_new", fossil='Gas', ax=axes[0, 0])
        #plot_break_even("Global", "Baseline-high", fossil='Gas', ax=ax1)
        
        #ax2 = plot_break_even("Global", "Baseline", fossil='Coal')
        plot_break_even("Global", "Baseline_new", fossil='Coal', ax=axes[0, 1])
        #plot_break_even("Global", "Baseline-high", fossil='Coal', ax=ax2)
            
        ax = plot_break_even_per_country("Baseline_new", fossil='Gas', ax=axes[1, 0])
        plot_break_even_per_country("Baseline_new", fossil='Coal', ax=axes[1, 1])
        plt.show()
        #fig.savefig(figures_directory + "Coal_gas_LCOE.svg", bbox_inches='tight')
    
        
       
    
#%% The grid resilience and the function of gas and coal
if "Global" in regs_to_print:
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(12, 4.8))
    df_loadband_global_coal = df_loadband[(df_loadband['Region']=='Global') & (df_loadband["Gas or coal"] == "Coal")]
    df_loadband_global_coal[df_loadband_global_coal['Scenario']=="Baseline_new"].plot.area(\
                               x="Year", ax=axes[0], linewidth=0, cmap="crest_r", legend=False)
    df_loadband_global_gas = df_loadband[(df_loadband['Region']=='Global') & (df_loadband["Gas or coal"] == "Gas")]
    df_loadband_global_gas[df_loadband_global_gas['Scenario']=="Baseline_new"].plot.area(\
                               x="Year", ax=axes[1], linewidth=0, cmap="crest_r")
    
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)
            #ax.tick_params(axis='y', which='major', pad=5)
            #ax.tick_params(axis='y', which=u'both',length=5)
            ax.set_xticks([2020, 2030, 2040, 2050, 2060], [2020, 2030, 2040, 2050, 2060], minor=False)   
            ax.get_yaxis().set_major_formatter( \
                mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            ax.set_ylabel("Generation (TWh/y)", fontsize=12)
    
    axes[0].set_ylim(0, 1500)
    axes[1].set_ylim(0, 320)
    axes[0].set_xlim(2020, 2060)
    axes[1].set_xlim(2020, 2060)
    axes[1].legend(fontsize=7)
    axes[0].set_title("Generation from coal")
    axes[1].set_title("Generation from gas")
    plt.show()

    
#%% Figure 4:
# Show maps with cheapest energy, show tipping point year (show distance from tipping)
if len(regs_to_print) > 50:
    # This only makes sense if you have all the separate world regions
    import geopandas
    
    # Get shapes of countries and exclude uninhabited areas
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    world = world[(world.pop_est>0) & (world.name!="Antarctica") & (world.name!="Fr. S. Antarctic Lands")]
    world = world.to_crs("ESRI:54030")
    
    E3ME_region = []
    for country in world.name:
        E3ME_region.append(country_to_E3ME_region70(country))
    world["E3ME region"] = E3ME_region
    
    # Per country, figure out which is the cheapest in year 2020 and 2030
    technologies = ['Coal', 'Gas', 'Hydro', "Nuclear", 'Onshore wind', "Offshore wind", 'Solar'] # Order same as df!
    
    fig, axes = plt.subplots(2, 2, figsize=(figsize_inches_two_column, figsize_inches_two_column*2/3), gridspec_kw={'hspace': -.25, 'wspace': -.23})

    df['E3ME region'] = [region[0]+1 for region in df['E3ME region']]
    axdict = {2020: (0,0), 2023: (0, 1), 2027: (1, 0), 2030: (1, 1)}   # A dictionary mapping the year to the location in overall graph

    for y in [2020, 2023, 2027, 2030]:
        cheapest = []
        
        for region in E3ME_region:
            df_slice = df[(df['E3ME region']==region) & (df['Year']==y)]
            prices = df_slice.iloc[0, 5:12]
            cheapest.append(technologies[np.nanargmin(prices)])
        world['Cheapest in '+ str(y)] = cheapest
        cheapest_tech = list(dict.fromkeys(cheapest))
        cheapest_tech.sort()
        cmap = ListedColormap([colours[i] for i in cheapest_tech], name='uniform_colours')
        g = world.plot(column='Cheapest in '+str(y), ax=axes[axdict[y]], \
                   legend=(True if y==2020 else False), legend_kwds={'loc': (0.06, -0.02), 'fontsize': 7, 'facecolor': 'none', 
                                                                     'framealpha': 0, 'handletextpad': 0.1},
                   cmap=cmap)
        axes[axdict[y]].set_title('Cheapest source in '+str(y))
        axes[axdict[y]].set_axis_off()
        
    world[["name", "iso_a3", "E3ME region", "Cheapest in 2020", "Cheapest in 2023", "Cheapest in 2027", "Cheapest in 2030"]].to_csv(source_data_directory + "Figure4.csv", index=False)

    
    # # Tipping point year
    # tipping_year = []
    # for region in E3ME_region:    
    #     for y in np.arange(2020, 2061):
    #         df_slice = df[(df['E3ME region']==region) & (df['Year']==y)]
    #         prices = df_slice.iloc[0, 5:10]
    #         if y == 2020:
    #             if prices.iloc[0] <= np.nanmin(prices):
    #                 tipping_year.append(y)
    #                 break
    #         if prices.iloc[0] <= np.nanmin(prices):
    #             tipping_year.append(y)
    #             break
    #         if y == 2060:
    #             tipping_year.append(np.nan)
    
    # world["Tipping year"] = tipping_year
    # #fig, ax = plt.subplots(1, 1)
    # g = world.plot(column='Tipping year', ax=axes[1,1], \
    #            legend=True, missing_kwds={'color': 'lightgrey'}, legend_kwds={'shrink': 0.7})
    # axes[1,1].set_title('Year solar becomes cheapest')
    # axes[1,1].set_axis_off()
    
    fig.tight_layout()
    fig.savefig(figures_directory + "map_compilation.svg", bbox_inches='tight')

        
        
    
