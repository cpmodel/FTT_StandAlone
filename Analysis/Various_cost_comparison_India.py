import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import get_output, get_metadata, save_fig, save_data



titles = {"ZWIC_India.csv": "Purchase cost MDT",
          "ZWFC_India.csv": 'Fuel cost MDT'}
units = {"ZWIC_India.csv": "Cost (1000$)",
          "ZWFC_India.csv": 'Cost ($/km)'}
# Load the CSV file (update the path if needed)
file_paths = ["ZWIC_India.csv", "ZWFC_India.csv"]

fig, axes = plt.subplots(1, 2, figsize=(8, 2.2))

for i_ax, file_path in enumerate(file_paths):
    
    df = pd.read_csv(file_path, skiprows=4)
    
    # Extract relevant columns
    df_filtered = df.iloc[:, [2, 16, 26]]  # Selecting category and years 2025, 2035
    df_filtered.columns = ["dimension2", "2025", "2035"]
    df_filtered[['Powertrain', 'Vehicle category']] = df['dimension2'].str.split(' ', n=1, expand=True)
    df_filtered.drop(columns=['dimension2'], inplace=True)
    
    # Convert cost columns to numeric values
    df_filtered["2025"] = pd.to_numeric(df_filtered["2025"], errors="coerce")
    df_filtered["2035"] = pd.to_numeric(df_filtered["2035"], errors="coerce")
    
    if i_ax == 0:
        df_filtered[["2025", "2035"]] /= 1000
    
    # Drop any rows with missing values
    df_filtered = df_filtered.dropna()
    
    # # Melt data for better visualization
    df = df_filtered.melt(id_vars=["Vehicle category", "Powertrain"], var_name="Year", value_name="Cost")
    
    # Define a function to create the plots
    def plot_vehicle_cost(data, category, ax):
        # Filter data for the specific category (MDT or HDT)
        df_filtered = data[data["Vehicle category"] == category].copy()
    
        # Create a custom order to introduce spacing between 2025 and 2035
        df_filtered["Category Group"] = df_filtered["Year"].astype(str) + " " + df_filtered["Powertrain"]
        category_order = ["2025 BEV", "2025 Diesel", " ", "2035 BEV", "2035 Diesel"]
        
        # Insert a spacer row
        spacer = pd.DataFrame({"Vehicle category": [category], "Powertrain": [""], "Year": [""], "Cost": [None], "Category Group": [" "]})
        df_final = pd.concat([df_filtered, spacer], ignore_index=True)
    
        # Ensure proper order
        df_final["Category Group"] = pd.Categorical(df_final["Category Group"], categories=category_order, ordered=True)
        
        sns.set_color_codes("pastel")
        
        # Plot the graph
        sns.barplot(data=df_final, y="Category Group", x="Cost",
                    palette={"BEV": "green", "Diesel": "grey"}, hue="Powertrain", ax=ax, dodge=False)
        ax.set_title(titles[file_path])
        ax.set_xlabel("")
        ax.set_ylabel(units[file_path])
        ax.legend(title="Powertrain", loc='center right')
        sns.despine(right=True, bottom=True)
    
    # # Create the figure with two subplots
    ax = axes[i_ax]
    
    # Plot for MDT and HDT
    plot_vehicle_cost(df, "MDT", ax)
    df_filtered.to_csv(f"Figure 11-{i_ax+1}.csv")

    
# Adjust layout
plt.tight_layout()
plt.show()

titles, fig_dir, tech_titles, _, _ = get_metadata()

save_fig(fig, fig_dir, "Figure 11 - Cost components BEV vs diesel India")


    
