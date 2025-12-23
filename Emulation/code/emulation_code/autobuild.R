
source('imported_code/UQ/Gasp.R') ### Created by James Salter
setwd('C:/Users/ib400/GitHub/FTT_StandAlone/Emulation/code/emulation_code')


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


#################################

## Set up data

#################################


#### DESIGN MATRIX ###########

file_path <- "C:/Users/ib400/Github/FTT_StandAlone/Emulation/"

### Load design matrix (inputs)
input_df <- read.csv(paste0(file_path, "data/scenarios/S3_scen_levels.csv"))
#input_df_rescaled <- read.csv(paste0(file_path, "data/scenarios/S3_scen_levels_rescaled.csv"))


##### OUTPUT MATRIX #########
csv_files <- list.files(path = "C:/Users/ib400/Github/FTT_StandAlone/Emulation/data/runs", 
                        pattern = "S3_.*\\.csv", full.names = T)

csv_files <- mixedsort(csv_files)

# Read each CSV file and store them in a list
dfs <- lapply(csv_files, read_csv)
full_output <- bind_rows(dfs)

########  EDIT VARS ##############################################

#full_output <- full_output[full_output$variable %in% c('MEWS'),] #'MEWP', 'MEWW', 'MEWE', 'MEWK'

##################################################################

#full_output_2 <- full_output
# Delete to save memory
dfs <- NULL

#############################

## Input rescaling

#############################



ranges <- list(
  learning_solar = list(min = -0.405, max = -0.233),
  learning_wind = list(min = -0.276, max = -0.112),
  lifetime_solar = list(min = 25, max = 35),
  lifetime_wind = list(min = 25, max = 35),
  lead_solar = list(min = 0.51, max = 1.5),
  lead_offshore = list(min = 2, max = 4),
  lead_onshore = list(min = 1, max = 2),
  lead_ccgt = list(min = 1, max = 2),
  lead_coal = list(min = 2, max = 3),
  lead_commission = list(min = 0, max = 1)
)


# List of columns to be rescaled
columns_to_rescale <- c('learning_solar',
                        'learning_wind',
                        'lifetime_solar',
                        'lifetime_wind',
                        'lead_solar',
                        'lead_offshore',
                        'lead_onshore',
                        'lead_ccgt',
                        'lead_coal',
                        'lead_commission')

# Copy input_df for rescaling
input_df_rescaled <- input_df

# Function for rescaling
rescale_column <- function(column, range) {
  (column - range$min) / (range$max - range$min)
}

# Rescale specified columns and overwrite the original columns
for (col in columns_to_rescale) {
  input_df_rescaled[[col]] <- rescale_column(input_df_rescaled[[col]], ranges[[col]])
}

# Function to invert the values in a column
invert_values <- function(values) {
  return(1 - values)
}

# Apply the inversion to a specific column in the dataframe
input_df_rescaled$learning_solar <- invert_values(input_df_rescaled$learning_solar)
input_df_rescaled$learning_wind <- invert_values(input_df_rescaled$learning_wind)

# check for values out of range
numeric_cols <- sapply(input_df_rescaled, is.numeric)
# Create a logical vector identifying rows with any numeric value outside 0-1
rows_outside <- apply(input_df_rescaled[, numeric_cols], 1, function(row) any(row < 0 | row > 1, na.rm = TRUE))
input_df_rescaled[rows_outside, ] ## should be 0

# save checked table
write.csv(input_df_rescaled, paste0(file_path, 'data/scenarios/S3_scen_levels_rescaled.csv'), row.names = FALSE)





#############################

## Single output emulator

############################
input_df_rescaled <- read.csv(paste0(file_path, 'data/scenarios/S3_scen_levels_rescaled.csv'))

# Set seed for function
seed_it = 5000

############## Function for building and saving emulators based on specific output_data
build_and_save_emulator <- function(output_data, key, seed = seed_it) {
  print(paste('Building emulator: ', key))
  # Merge with inputs for training
  train_data <- merge(input_df_rescaled, output_data[, c("scenario", "value")], by = "scenario", all = FALSE)
  
  # Add essential terms
  em_data <- data.frame(train_data[, 2:(ncol(train_data) - 1)],
                        Noise = runif(nrow(train_data), -1, 1),
                        value = train_data$value)
  
  # Build emulator with seed for training set
  set.seed(seed)
  em <- BuildGasp('value', em_data, mean_fn = 'step')
  
  # Save emulator
  saveRDS(em, file = paste0(file_path, "data/emulators/em_", key, ".rds"))
  
  # Generate validation plots
  png(paste0(file_path, "data/validation/plots/", key, ".png"), width = 1200, height = 900, res = 150)
  par(mfrow = c(5, 4), mar = c(4, 2, 2, 2))
  ValidateGasp(em, IndivPars = TRUE)
  dev.off()
  
  # Save regression analysis
  html_output <- paste0(file_path, "data/validation/models/", key, "_lm_summary.html")
  stargazer(em$lm$linModel, type = "html", out = html_output)
  print(paste('Emulator: ', key, ', validation plot and model saved'))
  return(em)  # Return the emulator object in case further analysis is needed
}



# Define general variables for output of interest
vars <- c( 'MEWK', 'MEWE', 'MEWP', 'MEWW', 'MEWS')
regions <- c('IN', 'GBL')
years <- c(2030, 2040, 2050)

# techs and subgroups
techs = list(
  # renewables
  c(short = 'solar', full = '19 Solar PV'),
  c(short = 'offshore', full = '18 Offshore'),
  c(short = 'onshore', full = '17 Onshore'),#,
  c(short = 'hydro_pump', full = '14 Pumped Hydro'),#,
  c(short = 'csp', full = '20 CSP'),
  c(short = 'biomass', full = '11 Biomass'),
  c(short = 'biomass', full = '12 Biomass + CCS'),
  # non-fossil based
  c(short = 'nuclear', full = '1 Nuclear'),#,
  c(short = 'hydro', full = '13 Large Hydro'),#,
  
  ### do we need to add to this for renew shares target?
  ### also take out nuke and hydro, geo?
  
  # fossil-based 
  c(short = 'gas', full = '7 CCGT'),
  c(short = 'coal', full = '3 Coal')
)
cap_techs <- techs[1:3]
renew_techs <- techs[1:7]
nonff_techs <- techs[1:9]


# Input df if needs reloading 
#input_df <- read.csv(paste0(file_path, "data/scenarios/S3_scen_levels.csv"))
#input_df_rescaled <- read.csv(paste0(file_path, "data/scenarios/S3_scen_levels_rescaled.csv"), row.names = 1)
# Loop through each combination of vars, techs, and regions
file_path <- "C:/Users/ib400/Github/FTT_StandAlone/Emulation/"



############## Execute for each var
for (var in vars) {
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
          for (tech in renew_techs){ # renew_techs
            if (region == 'GBL'){
              next
            } else {
                key <- paste0(var,"_", "nonff_", region, "_", end_year) #'renew_'
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









####### Build Bases - seperate and summed


# Reload the dictionary to store DataBasis_ftt and q for each combination
# DataBasis_path <- "C:/Users/ib400/Github/FTT_StandAlone/Emulation/data/designs/DataBasis_dict.rds"
# DataBasis_dict <- readRDS(DataBasis_path)
# DataBasis_dict <- list()
# 
# # Loop through each combination of vars, techs, and regions
# for (var in vars) {
#   # for (tech in techs) { 
#     # Adjust regions if var is MEWW to only perform once with region 'GBL'
#     region_list <- if (var == "MEWW") c("GBL") else regions
#     
#     for (region in region_list) {
#       
#       # Create a unique key based on the current combination
#       #key <- paste(var, tech["short"], region, sep = "_")
#       # summed var
#       key <- paste(var, region, sep = "_")
#       
#       
#       # Filter output_df for the specific combination of var, tech, and region
#       # output_df <- subset(full_output, 
#       #                     variable == var & 
#       #                       technology == tech["full"] &
#       #                       country_short == region)
#       # Block for summed variables e.g. emission
#       output_df <- full_output %>% subset(variable == var) %>%
#           group_by(scenario, year) %>%
#           summarise(year_value = sum(value, na.rm = TRUE))
# 
#       # If the subset is empty, skip to the next iteration
#       if (nrow(output_df) == 0) next
#       
#       # Convert to wide format with 'scenario' as columns, 'year' as rows, and 'value'
#       wide_data <- dcast(output_df, year ~ scenario, value.var = "year_value")
#       
#       # Reorder columns of wide_data to match input_df_rescaled scenarios
#       wide_data <- wide_data[, match(input_df_rescaled$scenario, colnames(wide_data))]
#       
#       # Create the basis object
#       DataBasis_ftt <- MakeDataBasis(as.matrix(wide_data))
#       
#       # Calculate the number of vectors to explain 99.99% of the variability
#       q <- ExplainT(DataBasis_ftt, vtot = 0.9999)
#       
#       # Store both DataBasis_ftt and q in the dictionary
#       DataBasis_dict[[key]] <- list(DataBasis_ftt = DataBasis_ftt, q = q)
#     }
#   }
# #}
# # Save dictionary
# saveRDS(DataBasis_dict, file = paste0("C:/Users/ib400/Github/FTT_StandAlone/Emulation/data/designs/DataBasis_dict.rds"))
# DataBasis_dict <- readRDS("C:/Users/ib400/Github/FTT_StandAlone/Emulation/data/designs/DataBasis_dict.rds")

#############################

### Build Bases Emulators

#############################

# # Define whether to split data for validation
# val <- "Y"  # Set to "Y" for validation split, "N" for full dataset training
# 
# # Initialize or reload the dictionary to store validation data
# # ValidationData_path <- paste0(file_path, "/data/designs/DataBasis_dict.rds")
# # ValidationData_dict <- readRDS(ValidationData_path)
# 
# ValidationData_dict <- list()
# 
# # TODO Create if statement for summed vars etc
# # Iterate over each key in DataBasis_dict
# for (key in names(DataBasis_dict[3])) {
#   
#   file_path <- "C:/Users/ib400/Github/FTT_StandAlone/Emulation/"
#   
#   # Extract var, tech, and region from the key
#   split_key <- strsplit(key, "_")[[1]]
#   var <- split_key[1]
#   #tech_short <- split_key[2]
#   region <- split_key[2]
#   
#   # Check if the current combination is in the selected lists
#   if (var %in% vars && tech_short %in% unlist(techs, recursive = TRUE) && region %in% regions) {
#     
#     # Retrieve DataBasis_ftt and q from DataBasis_dict
#     DataBasis_ftt <- DataBasis_dict[[key]]$DataBasis_ftt
#     q <- DataBasis_dict[[key]]$q
#     
#     # Data for coefficients, predicted by inputs, with Coeffs determining the projection
#     Coeffs <- Project(data = DataBasis_ftt$CentredField, 
#                       basis = DataBasis_ftt$tBasis[,1:q])
#     colnames(Coeffs)[1:q] <- paste("C", 1:q, sep = "")
#     
#     # Combine coefficients with design and add noise vector
#     input_coeffs <- data.frame(input_df_rescaled[,-1],  # Use the scaled version from before
#                                Noise = runif(nrow(input_df_rescaled), -1, 1), 
#                                Coeffs)
#     
#     ## Create train/test data if validation is needed
#     if (val == "Y") {
#       set.seed(321)
#       inds <- sample(1:nrow(input_df_rescaled), nrow(input_df_rescaled))
#       
#       # Calculate 90% for training and 10% for validation
#       train_size <- round(0.9 * nrow(input_df_rescaled))
#       train_inds <- inds[1:train_size]
#       val_inds <- inds[(train_size + 1):nrow(input_df_rescaled)]
#       
#       train_data <- input_coeffs[train_inds,]
#       val_data <- input_coeffs[val_inds,]
#       
#       # Store the validation data in the ValidationData_dict with the same key
#       ValidationData_dict[[key]] <- val_data
#     } else {
#       # Use the entire dataset for training if validation is not required
#       train_data <- input_coeffs
#     }
#     
#     # Build emulator on training data 
#     emulator <- BasisEmulators(train_data, q, mean_fn = 'step', maxdf = 5, training_prop = 1)
#     
#     # Save the emulator using the current var, tech_short, and region
#     saveRDS(emulator, file = paste0(file_path, "data/emulators/base_em_", key, ".rds"))
#     
#     # Save initial validation plots
#     if (val == "Y") {
#       for (i in seq(1, length(emulator), by = 1)) {
#         
#         png(paste0(file_path,"data/validation/validation_", key, i, ".png"), width = 1200, height = 900, res = 150)
#         par(mfrow = c(4,4), mar = c(4,2,2,2))
#         ValidateGasp(emulator[[i]], val_data)
#         dev.off()  # Close the graphics device
# 
#     } 
#   }
#   }
# }
# 
# # Save validation data
# saveRDS(ValidationData_dict, file = paste0(file_path, "/data/validation/ValidationData_dict.rds"))

      
