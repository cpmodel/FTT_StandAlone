import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define countries with full names
countries = {
    'Germany': '3 Germany (DE)',
    'United States': '34 USA (US)', 
    'China': '41 China (CN)',
    'India': '42 India (IN)'
}

# Define vehicle classes with lookup
vehicle_classes = ['Light Vehicles', 'Medium Duty', 'Heavy Duty']
vehicles_lookup = {'Light Vehicles': 'LCV', 'Medium Duty': 'MDT', 'Heavy Duty': 'HDT'}

# Define colors and linestyles including CNG/LPG
colors = {
    'Diesel': '#4B0082',
    'Petrol': '#FFA500',
    'BEV': '#228B22',
    'CNG/LPG': '#1E90FF'  # Adding blue for CNG/LPG
}

linestyles = {
    'Diesel': '--',
    'Petrol': '--',
    'BEV': '-',
    'CNG/LPG': '--'
}

def get_country_powertrains(country):
    """Return appropriate powertrains for each country"""
    if country == '41 China (CN)':
        return ['Diesel', 'CNG/LPG', 'BEV']
    return ['Diesel', 'Petrol', 'BEV']

def extract_vehicle_data(df, country_name, vehicle_class, powertrain):
    """Extract LCOF data for a specific vehicle type"""
    powertrain_mapping = {
        'Diesel': 'Diesel',
        'Petrol': 'Petrol',
        'BEV': 'BEV',
        'CNG/LPG': 'CNG/LPG'
    }
    
    tech_vehicle_class = vehicles_lookup[vehicle_class]
    
    country_mask = df['dimension'].str.contains(country_name, na=False, regex=False)
    country_df = df[country_mask]
    vehicle_mask = country_df['dimension2'].str.contains(f"{powertrain_mapping[powertrain]} {tech_vehicle_class}", na=False, regex=False)
    vehicle_row = country_df[vehicle_mask]

    if vehicle_row.empty:
        return None
    
    # Extract year columns
    year_columns = [col for col in df.columns if col.isdigit()]
    values = vehicle_row[year_columns].values[0]
    return values.astype(float)

def create_scaled_labeled_plots(df, countries):
    """Create a structured LCOF comparison plot with independent y-axis scaling"""
    fig, axes = plt.subplots(len(countries), len(vehicle_classes), figsize=(12, 14))
    
    # Store legend handles and labels
    legend_handles = []
    legend_labels = []

    # Extract year columns
    year_columns = [col for col in df.columns if col.isdigit()]
    years = [int(year) for year in year_columns]
    
    # Store legend handles and labels
    legend_handles = []
    legend_labels = []

    # Iterate through the grid and populate plots
    for i, (country_name, country_code) in enumerate(countries.items()):
        # Get appropriate powertrains for this country
        powertrains = get_country_powertrains(country_code)
        
        for j, vehicle_class in enumerate(vehicle_classes):
            ax = axes[i, j]
            y_min, y_max = float('inf'), float('-inf')

            for powertrain in powertrains:
                data = extract_vehicle_data(df, country_code, vehicle_class, powertrain)
                if data is not None:
                    start_idx = years.index(2025)
                    x_values = years[start_idx:]
                    y_values = data[start_idx:]
                    
                    y_min = min(y_min, np.min(y_values))
                    y_max = max(y_max, np.max(y_values))
                    
                    line = ax.plot(x_values, y_values, linestyle=linestyles[powertrain], 
                                 color=colors[powertrain], linewidth=2.5)
                    
                    # Store legend handles for unique powertrains
                    if powertrain not in legend_labels:
                        legend_handles.append(line[0])
                        legend_labels.append(powertrain)

            # Set grid and formatting
            ax.grid(True, linestyle='--', alpha=0.7, color='gray', linewidth=0.5)
            ax.set_axisbelow(True)
            
            if y_min < y_max:
                ax.set_ylim(y_min, y_max * 1.1)
            
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=12)
            
            ax.set_xticks([2025, 2030, 2035, 2040, 2045, 2050])
            ax.set_xlim(2024.99, 2051)
            
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Add labels
            if i == 0:
                ax.set_title(vehicle_class, fontsize=14, fontweight='bold', pad=10)
            if j == 0:
                ax.text(-0.4, 0.5, country_name, fontsize=14, fontweight='bold',
                       rotation=90, transform=ax.transAxes, 
                       verticalalignment='center')

    # Add tight layout before legend
    plt.tight_layout()

    # Add single legend at the bottom with more horizontal space
    fig.legend(legend_handles, legend_labels, 
              loc='center', bbox_to_anchor=(0.5, 0.02),
              ncol=4, fontsize=12,
              columnspacing=1.5,  # Increase space between legend columns
              handletextpad=0.5)  # Adjust space between handle and text

    # Add a single y-axis label for the entire figure
    fig.text(0.02, 0.5, 'Levelized Cost of Freight (USD/t-km)', rotation=90, 
             va='center', ha='center', fontsize=14, fontweight='bold')

    # Adjust layout to accommodate the legend and rotated labels
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.95, wspace=0.2, hspace=0.2)
    
    return fig

# Read and process the CSV
df = pd.read_csv("ZTLC.csv", delimiter=',', skiprows=4)
fig = create_scaled_labeled_plots(df, countries)
plt.show()