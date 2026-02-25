
# Load libraries ----

library(R.matlab)
library(ggplot2)
library(reshape2)
library(viridis)
library(cowplot)
library(fields)
library(dplyr)
library(scales)
library(reshape2)
library(tidyverse)
library(gtools)
library(Matrix)
library(patchwork)
library(tidyr)
library(lhs)
library(stargazer)
library(jsonlite)
#library(rstudioapi)
library(this.path)


# Set up working directory and source imported code ---------------------------------------------------------

script_dir <- this.path::this.dir()
wd <- dirname(dirname(dirname(script_dir)))
# set the working directory as FTT folder
setwd(wd)

## Source functions taken from other repositories
## Functons and scripts copied from ExeterUQ https://github.com/BayesExeter/ExeterUQ &
## https://github.com/JSalter90/UQ
# source code paths
uq_path <- "Emulation/code/emulation_code/imported_code/UQ/Gasp.R"
source(uq_path) ### Created by James Salter
# reset working dir
setwd(wd)

# Set up config file & paths ---------------------------------------------------

# import config
config_path <- "Emulation/code/config/config.json"
config <- jsonlite::fromJSON(config_path) 

# main paths TODO: are any unnecessary
data_path <- "Emulation/data"
input_path <- config$scen_levels_path
input_path_rescaled <- paste0(substr(input_path, 1, nchar(input_path)-4), "_rescaled.csv")

output_path <- paste0(data_path, "/runs")
# pattern for looping through runs
output_pattern <- paste0(config$scen_code, "_.*\\.csv")



# Data paths
emul_path <- "Emulation/data/emulators"
valid_plot_path <- "Emulation/data/validation/plots/"
valid_model_path <- "Emulation/data/validation/models/"


# Set up vars for emulators -----------------------------------------------

# output vars of interest
# if subsetting comment out and make new object preserving key
output_vars <- c('MEWK', # capacity
                 'MEWP', # electricity cost
                 'MEWS', # shares
                 'MEWW', # global capacity
                 'MEWE') # emissions

# regions
regions <- c('GBL', 
             'IN') # CN, US, RGN, RGS
years <- c(2030, 
           2040, 
           2050)

# techs and subgroups
techs = list(
  # renewables
  c(short = 'solar', full = '19 Solar PV'),
  c(short = 'onshore', full = '17 Onshore'),#,
  c(short = 'offshore', full = '18 Offshore'),
  c(short = 'hydro_pump', full = '14 Pumped Hydro'),#,
  c(short = 'csp', full = '20 CSP'),
  c(short = 'biomass', full = '11 Biomass'),
  c(short = 'biomass', full = '12 Biomass + CCS'),
  # non-fossil based
  c(short = 'nuclear', full = '1 Nuclear'),#,
  c(short = 'hydro', full = '13 Large Hydro'),#,
  # fossil-based 
  c(short = 'gas', full = '7 CCGT'),
  c(short = 'coal', full = '3 Coal')
)
cap_techs <- techs[c(1,2)]
renew_techs <- techs[1:7]
nonff_techs <- techs[1:9]


# Load input & output matrix -------------------------------------------------------

## load output
csv_files <- list.files(path = output_path, 
                        pattern = output_pattern, full.names = T)
csv_files <- mixedsort(csv_files)
dfs <- lapply(csv_files, read_csv) 
full_output <- bind_rows(dfs)


full_output <- full_output[full_output$variable %in% output_vars,] 

# Delete raw output for memory
dfs <- NULL

## load input
input_df <- read.csv(input_path)
#input_df_rescaled <- read.csv(input_path_rescaled)


# Normalise inputs --------------------------------------------------------

# extract policy  & technoeconomic cols
col_names <- names(input_df)
cols_pol <- col_names[grepl("pol$", col_names)]
cols_tech <- col_names[!grepl("pol$", col_names)]
# remove scenario id
cols_tech <- cols_tech[-1]

# load input parameter ranges and format
ranges <- lapply(config$ranges, function(x) list(min = x[1], max = x[2]))

# Copy input_df for rescaling
input_df_rescaled <- input_df

# Function for rescaling
rescale_column <- function(column, range) {
  (column - range$min) / (range$max - range$min)
}

# Rescale specified columns and overwrite the original columns
for (col in cols_tech) {
  input_df_rescaled[[col]] <- rescale_column(input_df_rescaled[[col]], ranges[[col]])
}


### Invert certain parameters for readability
# Function to invert the values in a column
invert_values <- function(values) {
  return(1 - values)
}

input_df_rescaled$learning_solar <- invert_values(input_df_rescaled$learning_solar)
input_df_rescaled$learning_wind <- invert_values(input_df_rescaled$learning_wind)
input_df_rescaled$cr_wind <- invert_values(input_df_rescaled$cr_wind)
input_df_rescaled$cr_solar <- invert_values(input_df_rescaled$cr_solar)


# check for values out of range
numeric_cols <- sapply(input_df_rescaled, is.numeric)
# Create a logical vector identifying rows with any numeric value outside 0-1
rows_outside <- apply(input_df_rescaled[, numeric_cols], 1, function(row) any(row < 0 | row > 1, na.rm = TRUE))
input_df_rescaled[rows_outside, ] ## should be 0
# Error message if ranges out of bounds
if(nrow(input_df_rescaled[rows_outside, ]) != 0) {
  stop("Error: Rows outside bounds detected!")
}

# save checked table
write.csv(input_df_rescaled, input_path_rescaled, row.names = FALSE)



# Build single year emulator ----------------------------------------------

# Reload if needed
#input_df_rescaled <- read.csv(input_path_rescaled)

# Set seed for function
seed_it = 5000

############## Function for building and saving emulators
build_and_save_emulator <- function(output_data, key, seed = seed_it) {
  print(paste('Building emulator: ', key))
  # Merge with inputs for training
  train_data <- merge(input_df_rescaled, output_data[, c("scenario", "value")], by = "scenario", all = FALSE)
  
  # Add essential terms
  em_data <- data.frame(train_data[, 2:(ncol(train_data) - 1)],
                        Noise = runif(nrow(train_data), -1, 1),
                        value = train_data$value)
  
  # Build emulator with seed for training set
  set.seed(seed_it)
  em <- BuildGasp('value', em_data, mean_fn = 'step')
  
  # Save emulator
  saveRDS(em, file = paste0(emul_path, "em_", key, ".rds"))
  
  # Generate validation plots
  png(paste0(valid_plot_path, key, ".png"), width = 1200, height = 900, res = 150)
  par(mfrow = c(5, 4), mar = c(4, 2, 2, 2))
  ValidateGasp(em, IndivPars = TRUE)
  dev.off()
  
  # Save regression analysis
  html_output <- paste0(valid_model_path, key, "_lm_summary.html")
  stargazer(em$lm$linModel, type = "html", out = html_output)
  print(paste('Emulator: ', key, ', validation plot and model saved'))
  return(em)  # Return the emulator object in case further analysis is needed
}



# Execute -----------------------------------------------------------------

# Loop through vars
for (var in output_vars) {
  for (end_year in years){
    if (var == 'MEWW'){
      for (tech in cap_techs){
        key <- paste0(var,"_", tech["short"], "_", "GBL", "_", end_year)
        output_data <- subset(full_output, year == end_year &
                                variable == var &
                                technology == tech['full'])
        build_and_save_emulator(output_data, key)
      }
  } else if (var == 'MEWE'){
      for (region in regions){
        key <- paste0(var,"_", region, "_", end_year)
        if (region == 'GBL'){
          # Filter output
          output_data <- subset(full_output, year == end_year &
                                  variable == var) %>%
            group_by(scenario, year) %>%
            summarise(value = sum(value, na.rm = TRUE))
            # Save based on key
            build_and_save_emulator(output_data, key)
        } else {
              # Filter output
              output_data <- subset(full_output, year == end_year &
                                    variable == var &
                                    country_short == region) %>%
              group_by(scenario, year) %>%
              summarise(value = sum(value, na.rm = TRUE))
              # Save based on key
              build_and_save_emulator(output_data, key)
        }
      }
    } else if (var %in% c('MEWK', 'MEWG', 'MEWC')){
        for (region in regions){
          for (tech in cap_techs){
            if (region == 'GBL'){
              next
              
              } else {
                  key <- paste0(var,"_", tech["short"],"_", region, "_", end_year)
                  # Filter output
                  output_data <- subset(full_output, year == end_year &
                                          variable == var &
                                          technology == tech['full'] &
                                          country_short == region)
                  # Saved based on key
                  build_and_save_emulator(output_data, key)
          }
        }  
      } 
    } else if (var == 'MEWS'){
        for (region in regions){
            if (region == 'GBL'){
              next
            } else {
                key <- paste0(var,"_", "renew_", region, "_", end_year) #'renew_'
                # Filter output
                output_data <- subset(full_output, year == end_year &
                                        variable == var &
                                        technology %in% unlist(renew_techs, recursive = TRUE) &
                                        country_short == region) %>%

                  group_by(scenario, year) %>%
                  summarise(value = sum(value, na.rm = TRUE))
                
                # Saved based on key
                build_and_save_emulator(output_data, key)
      }
    }
  } else if (var == 'MEWP'){
      for (region in regions){
        if (region == 'GBL'){
          next
          
        } else {
        tech <- '8 Electricity'
        key <- paste0(var,"_", "elec_price_", region, "_", end_year)
        # Filter output
        output_data <- subset(full_output, year == end_year &
                                variable == var &
                                technology == tech &
                                country_short == region)
        # Saved based on key
        
        build_and_save_emulator(output_data, key)
    }
  }
  }
  }
}









