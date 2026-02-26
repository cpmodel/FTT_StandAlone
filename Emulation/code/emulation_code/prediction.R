
# Load libraries ----------------------------------------------------------

library(ggplot2)
library(tidyr)


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


# Data paths
emul_path <- "Emulation/data/emulators"

## Output paths
output_dir <- paste0(data_path, "/predictions/")

# Scenario file names
output_scenarios_in <- "input_dfs_IN_lhs.RData" 
output_scenarios_gbl <- "input_dfs_GBL_lhs.RData" 

# Prediction file names
out_in_preds <- "preds_IN_lhs.csv"
out_gbl_preds <- "preds_GBL_lhs.csv"


# Load input data and basic functions ---------------------------------------------------------

input_df_rescaled <- read.csv(input_path_rescaled)

# Function for rescaling used in loop below
rescale_column <- function(column, range) {
  (column - range$min) / (range$max - range$min)
}

# Extract policy variables 
pol_params <- names(input_df_rescaled[2:16])
# Extract techecon variables 
tech_params <- names(input_df_rescaled[17:30])




# Emulator Selections  ------------------------------------------------

# option for using emulator mean or sampling from uncertainty, mean by default
emulator_uncertainty <- "NO" # "YES"

### Global
# List of emulator selections to run
gbl_emulator_selections <- list("MEWE_GBL_2030", "MEWE_GBL_2050")


### India
# List of emulator selections to run
in_emulator_selections <- list("MEWK_solar_IN_2030", "MEWK_solar_IN_2040", "MEWK_solar_IN_2050", 
                            "MEWK_onshore_IN_2030", "MEWK_onshore_IN_2040", "MEWK_onshore_IN_2050",
                            "MEWS_renew_IN_2030", "MEWS_renew_IN_2040", "MEWS_renew_IN_2050",
                            "MEWE_IN_2030", "MEWE_IN_2040", "MEWE_IN_2050", 
                            "MEWP_elec_price_IN_2030", "MEWP_elec_price_IN_2040", "MEWP_elec_price_IN_2050"
)


# Scenario design - GLOBAL --------------------------------------------------

# Define grid values
grid_vals <- c(0, 0.5, 1)

# Create a matrix with 3 rows, each filled with 0, 0.5, and 1 respectively
values <- matrix(rep(c(0, 0.5, 1), each = length(pol_params)), 
                 nrow = 3, byrow = TRUE)

# Convert to dataframe and assign column names
pol_df  <- as.data.frame(values)
names(pol_df) <- pol_params

n_iterations <- 10000

# Define which columns should use LHS, ALL BY DEFAULT
lhs_cols <- tech_params
# c("elec_demand", "lead_commission", "lead_solar", 
#               "lead_onshore", "discr", "cr_wind", "cr_solar")

# Scenario design - INDIA ----------------------------------------------

# Create a base row with all 0s
base_row <- as.data.frame(matrix(0, nrow = 1, ncol = length(pol_params)))
names(base_row) <- pol_params

# edit policy vars that will remain fixed
base_row[, c( #"CN_phase_pol", "CN_price_pol", "CN_cp_pol", 
  "US_price_pol")] <- 0.5

# Define grid values for India
grid_vals <- c(0, 0.5, 1)

# Create a grid of all combinations for the two variables
grid <- expand.grid(IN_phase_pol = grid_vals,
                    IN_price_pol = grid_vals, 
                    IN_cp_pol = grid_vals)

# Replicate base_row for each combination in the grid
pol_df <- base_row[rep(1, nrow(grid)), ]

# Overwrite the values for IN_phase_pol and IN_price_pol with the grid
pol_df$IN_phase_pol <- grid$IN_phase_pol
pol_df$IN_price_pol <- grid$IN_price_pol
pol_df$IN_cp_pol <- grid$IN_cp_pol

n_iterations <- 10000

# Define which columns should use LHS, ALL BY DEFAULT, rest will be normally distributed
lhs_cols <- tech_params
#c("elec_demand", "lead_commission", "lead_solar",
# "lead_onshore", "discr", "cr_wind", "cr_solar")


# Scenario creation - GLOBAL ------------------------------------------------

# Initialize list to store each row's data frame of samples
input_dfs <- list()

for (i in 1:nrow(pol_df)) {
  
  fixed_params <- pol_df[i, ]
  varied_columns <- setdiff(names(input_df_rescaled[, 2:length(names(input_df_rescaled))]), names(fixed_params))
  
  ## Split columns
  lhs_use <- intersect(lhs_cols, varied_columns)
  normal_use <- setdiff(varied_columns, lhs_cols)
  
  ## Prepare empty list
  lhs_sample <- NULL
  normal_sample <- NULL
  
  # LHS sampling for specific columns
  if (length(lhs_use) > 0) {
    var_min_lhs <- rep(0, length(lhs_use))
    var_max_lhs <- rep(1, length(lhs_use))
    
    lhs_matrix <- randomLHS(n_iterations, length(lhs_use))
    
    lhs_sample <- as.data.frame(lhs_matrix)
    colnames(lhs_sample) <- lhs_use
  }
  
  # Normal distribution for the rest
  if (length(normal_use) > 0) {
    var_min_norm <- rep(0, length(normal_use))
    var_max_norm <- rep(1, length(normal_use))
    mean_values <- (var_min_norm + var_max_norm) / 2
    sd_values <- (var_max_norm - var_min_norm) / 4
    
    normal_sample <- as.data.frame(
      sapply(seq_along(normal_use), function(j) {
        sampled_values <- rnorm(n_iterations, mean = mean_values[j], sd = sd_values[j])
        pmin(pmax(sampled_values, 0), 1)
      })
    )
    colnames(normal_sample) <- normal_use
  }
  
  # Combine LHS + normal samples into one data frame (original column order) if varying
  if (length(normal_use) > 0) {
    varied_sample <- cbind(lhs_sample, normal_sample)[, varied_columns]
  }
  else {
    varied_sample <- lhs_sample
  }
  
  # Combine with fixed parameters
  input_df_sample <- cbind(
    as.data.frame(matrix(rep(unlist(fixed_params), n_iterations), nrow = n_iterations, byrow = TRUE)),
    varied_sample
  )
  
  colnames(input_df_sample)[1:length(fixed_params)] <- names(fixed_params)
  
  input_dfs[[paste0("id_", i)]] <- input_df_sample
}


# Save input scenarios
saveRDS(input_dfs, paste0(output_dir, output_scenarios_gbl))

## Input Sample distribution plot
# Pick one of the generated data frames
df_check <- input_dfs[["id_1"]]

# Melt into long format for faceting
df_long <- df_check %>%
  pivot_longer(cols = all_of(varied_columns), names_to = "variable", values_to = "value")

# Plot histograms for each variable
p <- ggplot(df_long, aes(x = value)) +
  geom_histogram(binwidth = 0.02, boundary = 0, fill = "skyblue", color = "black") +
  facet_wrap(~variable, scales = "free") +
  theme_minimal() +
  labs(title = "Distribution of Sampled Variables (India)", x = "Value", y = "Count")
# Save to figures folder
ggsave(paste0(data_path, "/figures/scen_input_dist_GBL.png"), p, 
       width = 8, height = 6, dpi = 100)

# Scenario creation - INDIA -------------------------------------------------

# Initialize storage
input_dfs <- list()

for (i in 1:nrow(pol_df)) {
  
  fixed_params <- pol_df[i, ]
  varied_columns <- setdiff(names(input_df_rescaled[, 2:length(names(input_df_rescaled))]), names(fixed_params))
  
    ## Split columns
    lhs_use <- intersect(lhs_cols, varied_columns)
    normal_use <- setdiff(varied_columns, lhs_cols)
    
    ## Prepare empty list
    lhs_sample <- NULL
    normal_sample <- NULL
    
    # LHS sampling for specific columns
    if (length(lhs_use) > 0) {
      var_min_lhs <- rep(0, length(lhs_use))
      var_max_lhs <- rep(1, length(lhs_use))
      
      lhs_matrix <- randomLHS(n_iterations, length(lhs_use))
      
      lhs_sample <- as.data.frame(lhs_matrix)
      colnames(lhs_sample) <- lhs_use
    }
    
    # Normal distribution for the rest
    if (length(normal_use) > 0) {
      var_min_norm <- rep(0, length(normal_use))
      var_max_norm <- rep(1, length(normal_use))
      mean_values <- (var_min_norm + var_max_norm) / 2
      sd_values <- (var_max_norm - var_min_norm) / 4
      
      normal_sample <- as.data.frame(
        sapply(seq_along(normal_use), function(j) {
          sampled_values <- rnorm(n_iterations, mean = mean_values[j], sd = sd_values[j])
          pmin(pmax(sampled_values, 0), 1)
        })
      )
      colnames(normal_sample) <- normal_use
    }
    
    # Combine LHS + normal samples into one data frame (original column order) if varying
    if (length(normal_use) > 0) {
    varied_sample <- cbind(lhs_sample, normal_sample)[, varied_columns]
    }
    else {
      varied_sample <- lhs_sample
    }
  
  # Combine with fixed parameters
  input_df_sample <- cbind(
    as.data.frame(matrix(rep(unlist(fixed_params), n_iterations), nrow = n_iterations, byrow = TRUE)),
    varied_sample
  )
  
  colnames(input_df_sample)[1:length(fixed_params)] <- names(fixed_params)
  
  input_dfs[[paste0("id_", i)]] <- input_df_sample
}


# Save file
saveRDS(input_dfs, file = paste0(output_dir, output_scenarios_in))


## Input Sample distribution plot
# Pick one of the generated data frames
df_check <- input_dfs[["id_1"]]

# Melt into long format for faceting
df_long <- df_check %>%
  pivot_longer(cols = all_of(varied_columns), names_to = "variable", values_to = "value")

# Plot histograms for each variable
p <- ggplot(df_long, aes(x = value)) +
      geom_histogram(binwidth = 0.02, boundary = 0, fill = "skyblue", color = "black") +
      facet_wrap(~variable, scales = "free") +
      theme_minimal() +
      labs(title = "Distribution of Sampled Variables (India)", x = "Value", y = "Count")
# Save to figures folder
ggsave(paste0(data_path, "/figures/scen_input_dist_IN.png"), p, 
              width = 8, height = 6, dpi = 100)



# Emulator predictions - GLOBAL ---------------------------------------------

# Reload fileinput_dfs <- readRDS(paste0(output_dir, output_scenarios_gbl))
#input_dfs <- readRDS("C:/Users/ib400/Github/FTT_StandAlone/Emulation/data/predictions/input_dfs_GBL_emiss.RData")

# combine into df
all_inputs_df <- do.call(rbind, input_dfs)

# Optionally reset row names
rownames(all_inputs_df) <- NULL

# Convert to df (only type allowed by emulators)
all_inputs_mat <- data.frame(all_inputs_df)

# List to store all results
all_results <- list()

# Loop over each emulator
for (selection in gbl_emulator_selections) {
  
  # Load emulator
  emulator_path <- paste0(emul_path, "/em_", selection, ".rds")
  emulator <- readRDS(emulator_path)
  
  # Extract year from emulator name
  year <- substr(selection, nchar(selection) - 3, nchar(selection))  # last 4 characters
  
  # Loop over each input set
  for (id_name in names(input_dfs)) {
    
    input_df <- input_dfs[[id_name]]
    
    # Predict
    pred <- PredictGasp(input_df, emulator)
    pred_mean <- pred$mean
    
    # option to sample from emulator uncertainty
    samp <- rnorm(length(pred$mean), mean = pred$mean, sd = pred$sd)
    
    if (emulator_uncertainty == "YES") {
      result_df <- cbind(input_df, prediction = samp)
    
      } else {
        # Combine with inputs
        result_df <- cbind(input_df, prediction = pred_mean)
    }
    
    
    # Add metadata
    result_df$emulator   <- selection
    result_df$id         <- id_name
    result_df$year       <- year
    result_df$sample_id  <- seq_len(nrow(input_df))
    
    # Store result
    all_results[[paste0(selection, "_", id_name)]] <- result_df
  }
}

# Combine all into a single data.frame
final_results_df <- do.call(rbind, all_results)

# Optional: reset rownames
rownames(final_results_df) <- NULL

# Save
write.csv(final_results_df, 
          file = paste0(output_dir, out_gbl_preds),
          row.names = F)



# Emulator predictions - INDIA --------------------------------------------

# Reload file if needed
#input_dfs <- readRDS(paste0(output_dir, output_scenarios_in))

# combine into df
all_inputs_df <- do.call(rbind, input_dfs)

# Optionally reset row names
rownames(all_inputs_df) <- NULL

# Convert to df (only type allowed by emulators)
all_inputs_mat <- data.frame(all_inputs_df)

# List to store all results
all_results <- list()

# Loop over each emulator
for (selection in in_emulator_selections) {
  
  # Load emulator
  emulator_path <- paste0(emul_path, "/em_", selection, ".rds")
  emulator <- readRDS(emulator_path)
  
  # Extract year from emulator name
  year <- substr(selection, nchar(selection) - 3, nchar(selection))  # last 4 characters
  
  # Loop over each input set
  for (id_name in names(input_dfs)) {
    
    input_df <- input_dfs[[id_name]]
    
    # Predict
    pred <- PredictGasp(input_df, emulator)
    pred_mean <- pred$mean
    
    # option to sample from emulator uncertainty
    samp <- rnorm(length(pred$mean), mean = pred$mean, sd = pred$sd)
    
    if (emulator_uncertainty == "YES") {
      result_df <- cbind(input_df, prediction = samp)
      
    } else {
      # Combine with inputs
      result_df <- cbind(input_df, prediction = pred_mean)
    }
    
    # Add metadata
    result_df$emulator   <- selection
    result_df$id         <- id_name
    result_df$year       <- year
    result_df$sample_id  <- seq_len(nrow(input_df))
    
    # Store result
    all_results[[paste0(selection, "_", id_name)]] <- result_df
  }
}


# Combine all into a single data.frame
final_results_df <- do.call(rbind, all_results)

# Optional: reset rownames
rownames(final_results_df) <- NULL

# Save 
write.csv(final_results_df, 
          file = paste0(output_dir, out_in_preds),
          row.names = F)







