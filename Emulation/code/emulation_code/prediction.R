
####
library(ggplot2)
library(tidyr)


## Retrieve functions taken from other repositories
## Functons and scripts copied from ExeterUQ https://github.com/BayesExeter/ExeterUQ &
## https://github.com/JSalter90/UQ
#setwd('C:/Users/ib400/GitHub/FTT_StandAlone/Emulation/code/emulation_code/imported_code')
source('C:/Users/ib400/GitHub/FTT_StandAlone/Emulation/code/emulation_code/imported_code/UQ/Gasp.R') ### Created by James Salter
setwd('C:/Users/ib400/GitHub/FTT_StandAlone/Emulation/code/emulation_code')


#DataBasis_dict <- readRDS(paste0("C:/Users/ib400/Github/FTT_StandAlone/Emulation/data/designs/DataBasis_dict.rds"))
input_df <- read.csv("C:/Users/ib400/Github/FTT_StandAlone/Emulation/data/scenarios/S3_scen_levels.csv")
input_df_rescaled <- read.csv("C:/Users/ib400/Github/FTT_StandAlone/Emulation/data/scenarios/S3_scen_levels_rescaled.csv")


# Function for rescaling used in loop below
rescale_column <- function(column, range) {
  (column - range$min) / (range$max - range$min)
}



#############################

###### New design

#############################

# Extract policy variables 
pol_params <- names(input_df_rescaled[2:16])

# Create a base row with all 0s
base_row <- as.data.frame(matrix(0, nrow = 1, ncol = length(pol_params)))
names(base_row) <- pol_params

amb_row <- base_row
amb_row[, c( #"CN_phase_pol", "CN_price_pol", "CN_cp_pol", 
            "US_price_pol")] <- 0.5


## India policy comparison
# Define grid values
grid_vals <- c(0, 0.5, 1)

# Create a grid of all combinations for the two variables
grid <- expand.grid(IN_phase_pol = grid_vals,
                    IN_price_pol = grid_vals, 
                    IN_cp_pol = grid_vals)

# Replicate amb_row for each combination in the grid
pol_df <- amb_row[rep(1, nrow(grid)), ]

# Overwrite the values for IN_phase_pol and IN_price_pol with the grid
pol_df$IN_phase_pol <- grid$IN_phase_pol
pol_df$IN_price_pol <- grid$IN_price_pol
pol_df$IN_cp_pol <- grid$IN_cp_pol

##########################################################################


n_iterations <- 10000
sampling_method <- "lhs"  # Options: "lhs", "normal", or "set"

# Initialize storage
input_dfs <- list()

# Define which columns should use LHS
lhs_cols <- c("elec_demand", "lead_commission", "lead_solar", 
              "lead_onshore", "discr", "cf_wind", "cf_solar")

for (i in 1:nrow(pol_df)) {
  
  fixed_params <- pol_df[i, ]
  varied_columns <- setdiff(names(input_df[, 2:length(names(input_df))]), names(fixed_params))
  
  if (sampling_method == "set") {
    
    varied_sample <- as.data.frame(matrix(0.5, nrow = 1, ncol = length(varied_columns)))
    colnames(varied_sample) <- varied_columns
    
  } else if (sampling_method == "normal") {
    
    var_min <- rep(0, length(varied_columns))
    var_max <- rep(1, length(varied_columns))
    mean_values <- (var_min + var_max) / 2
    sd_values <- (var_max - var_min) / 4
    
    varied_sample <- as.data.frame(
      sapply(seq_along(varied_columns), function(j) {
        sampled_values <- rnorm(n_iterations, mean = mean_values[j], sd = sd_values[j])
        pmin(pmax(sampled_values, 0), 1)
      })
    )
    colnames(varied_sample) <- varied_columns
    
  } else if (sampling_method == "lhs") {
    
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
    
    # Combine LHS + normal samples into one data frame (original column order)
    varied_sample <- cbind(lhs_sample, normal_sample)[, varied_columns]
    
  } else {
    stop("Invalid sampling method. Choose 'lhs', 'normal', or 'set'.")
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
saveRDS(input_dfs, file = "C:/Users/ib400/Github/FTT_StandAlone/Emulation/data/predictions/input_dfs_IN_polcomp_grid.RData")

# Reload file if needed
#input_dfs <- readRDS("C:/Users/ib400/Github/FTT_StandAlone/Emulation/data/predictions/input_dfs_IN_polcomp_grid.RData")

# combine into df
all_inputs_df <- do.call(rbind, input_dfs)

# Optionally reset row names
rownames(all_inputs_df) <- NULL

# Convert to df (only type allowed by emulators)
all_inputs_mat <- data.frame(all_inputs_df)

# List of emulator selections to run
emulator_selections <- list("MEWK_solar_IN_2030", "MEWK_solar_IN_2040", "MEWK_solar_IN_2050", 
                            "MEWK_onshore_IN_2030", "MEWK_onshore_IN_2040", "MEWK_onshore_IN_2050",
                            "MEWS_renew_IN_2030", "MEWS_renew_IN_2040", "MEWS_renew_IN_2050",
                            "MEWE_IN_2030", "MEWE_IN_2040", "MEWE_IN_2050", 
                            "MEWP_elec_price_IN_2030", "MEWP_elec_price_IN_2040", "MEWP_elec_price_IN_2050"
)



############## Single year emulators

# List to store all results
all_results <- list()

# Loop over each emulator
for (selection in emulator_selections) {
  
  # Load emulator
  emulator_path <- paste0("C:/Users/ib400/Github/FTT_StandAlone/Emulation/data/emulators/em_", selection, ".rds")
  emulator <- readRDS(emulator_path)
  
  # Extract year from emulator name
  year <- substr(selection, nchar(selection) - 3, nchar(selection))  # last 4 characters
  
  # Loop over each input set
  for (id_name in names(input_dfs)) {
    
    input_df <- input_dfs[[id_name]]
    
    # Predict
    pred <- PredictGasp(input_df, emulator)
    #pred_mean <- pred$mean
    
    # Instead, sample from emulator uncertainty
    samp <- rnorm(length(pred$mean), mean = pred$mean, sd = pred$sd)
    
    # Combine with inputs
    result_df <- cbind(input_df, prediction = samp) #pred_mean
    
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

# Save to df
write.csv(final_results_df, 
          file = "C:/Users/ib400/Github/FTT_StandAlone/Emulation/data/predictions/IN_polcomp_lead_grid_2.csv",
          row.names = F)


####################################################

## Global uncertainty analysis 

####################################################

# Define grid values
grid_vals <- c(0, 0.5, 1)

pol_params <- names(input_df[2:16])

# Create a matrix with 3 rows, each filled with 0, 0.5, and 1 respectively
values <- matrix(rep(c(0, 0.5, 1), each = length(pol_params)), 
                 nrow = 3, byrow = TRUE)

# Convert to dataframe and assign column names
pol_df  <- as.data.frame(values)
names(pol_df) <- pol_params


#######################

### Sampling

#######################

n_iterations <- 20000
sampling_method <- "normal"  # Options: "lhs" or "normal"

# Initialize list to store each row's data frame of samples
input_dfs <- list()

# Loop through each row in plot_grid
for (i in 1:nrow(pol_df)) {
  # Extract fixed parameters for the current row
  fixed_params <- pol_df[i, ]
  
  # Identify which columns in input_df need to be varied
  varied_columns <- setdiff(names(input_df[, 2:length(names(input_df))]), names(fixed_params))
  
  # Generate samples based on the selected sampling method
  if (sampling_method == "set") {
    # Generate a data frame where all values are 0.5
    varied_sample <- as.data.frame(matrix(0.5, nrow = 1, ncol = length(varied_columns)))
    colnames(varied_sample) <- varied_columns
    
  } else if (sampling_method == "normal") {
    # Normal Distribution Sampling
    # Define ranges (mean and sd) for the varied columns
    # Replace this logic with actual ranges for your data
    var_min <- rep(0, length(varied_columns))  # Example: min = 0 for all variables
    var_max <- rep(1, length(varied_columns))  # Example: max = 1 for all variables
    mean_values <- (var_min + var_max) / 2     # Midpoint of the range
    sd_values <- (var_max - var_min) / 4      
    
    # Generate samples from a normal distribution
    varied_sample <- as.data.frame(
      sapply(seq_along(varied_columns), function(j) {
        sampled_values <- rnorm(n_iterations, mean = mean_values[j], sd = sd_values[j])
        # Clip the values to ensure they are within the [0, 1] range
        sampled_values <- pmin(pmax(sampled_values, 0), 1)
        
        return(sampled_values)
      })
    )
    
    # Set column names
    colnames(varied_sample) <- varied_columns
    
    
  } else {
    stop("Invalid sampling method. Choose 'lhs' or 'normal'.")
  }
  
  # Combine fixed parameters and varied sample for this row
  input_df_sample <- cbind(
    as.data.frame(matrix(rep(unlist(fixed_params), n_iterations), nrow = n_iterations, byrow = TRUE)),
    varied_sample
  )
  
  # Set column names for the fixed parameters
  colnames(input_df_sample)[1:length(fixed_params)] <- names(fixed_params)
  
  # Add to dictionary with row number as the key
  input_dfs[[paste0("id_", i)]] <- input_df_sample
}



# Save file
saveRDS(input_dfs, file = "C:/Users/ib400/Github/FTT_StandAlone/Emulation/data/predictions/input_dfs_GBL_emiss.RData")

# Reload file
#input_dfs <- readRDS("C:/Users/ib400/Github/FTT_StandAlone/Emulation/data/predictions/input_dfs_GBL_emiss.RData")

# combine into df
all_inputs_df <- do.call(rbind, input_dfs)

# Optionally reset row names
rownames(all_inputs_df) <- NULL

# Convert to df (only type allowed by emulators)
all_inputs_mat <- data.frame(all_inputs_df)

# List of emulator selections to run
emulator_selections <- list("MEWE_GBL_2030", "MEWE_GBL_2050")



############## Single year emulators

# List to store all results
all_results <- list()

# Loop over each emulator
for (selection in emulator_selections) {
  
  # Load emulator
  emulator_path <- paste0("C:/Users/ib400/Github/FTT_StandAlone/Emulation/data/emulators/em_", selection, ".rds")
  emulator <- readRDS(emulator_path)
  
  # Extract year from emulator name
  year <- substr(selection, nchar(selection) - 3, nchar(selection))  # last 4 characters
  
  # Loop over each input set
  for (id_name in names(input_dfs)) {
    
    input_df <- input_dfs[[id_name]]
    
    # Predict
    pred <- PredictGasp(input_df, emulator)
    pred_mean <- pred$mean
    
    # Combine with inputs
    result_df <- cbind(input_df, prediction = pred_mean)
    
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

# Save to df
write.csv(final_results_df, 
          file = "C:/Users/ib400/Github/FTT_StandAlone/Emulation/data/predictions/GBL_emissions_amb.csv",
          row.names = F)


###################


## INPUT PLOT FOR APPENDIX

# Pick one of the generated data frames
df_check <- input_dfs[["id_1"]]

# Melt into long format for faceting
df_long <- df_check %>%
  pivot_longer(cols = all_of(varied_columns), names_to = "variable", values_to = "value")

# Plot histograms for each variable
ggplot(df_long, aes(x = value)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  facet_wrap(~variable, scales = "free") +
  theme_minimal() +
  labs(title = "Distribution of Sampled Variables", x = "Value", y = "Count")





# 
# ########### Function for bases emulators
# 
# # Function to process a batch of input_dfs and save results
# process_batch <- function(batch, batch_number, output_file) {
#   batch_results <- list()
#   
#   for (selection in emulator_selections) {
#     emulator <- readRDS(paste0("C:/Users/ib400/Github/FTT_StandAlone/Emulation/data/emulators/em_", selection, ".rds"))
#     DataBasis_ftt <- DataBasis_dict[[selection]]$DataBasis_ftt
#     q <- DataBasis_dict[[selection]]$q
#     
#     for (j in range(1:length(batch))) {
#       id <- batch[[j]]
#       input_df <- input_dfs[[id]]
#       sample_id <- paste0(1:nrow(input_df))
#       
#       preds_new_basis <- BasisPredGasp(input_df, emulator)
#       preds_recon <- matrix(0, nrow = nrow(DataBasis_ftt$tBasis), ncol = nrow(preds_new_basis$Expectation))
#       
#       for (i in 1:nrow(preds_new_basis$Expectation)) {
#         preds_recon[, i] <- Recon(preds_new_basis$Expectation[i,], DataBasis_ftt$tBasis[, 1:q]) + DataBasis_ftt$EnsembleMean
#       }
#       
#       recon_df <- as.data.frame(t(preds_recon))
#       colnames(recon_df) <- seq(2010, 2050)
#       recon_df$emulator <- selection
#       recon_df$id <- id
#       recon_df$sample_id <- sample_id
#       
#       # Melt to long format with `Year` and `Value`
#       recon_long <- melt(recon_df, id.vars = c("sample_id", "id", "emulator"), variable.name = "Year", value.name = "Value")
#       # Convert `Year` to numeric
#       recon_long$Year <- as.numeric(as.character(recon_long$Year))
#       # Convert `sample_id` to numeric by extracting the number part
#       recon_long$sample_id <- as.numeric(gsub("sample_", "", recon_long$sample_id))
#       # Sort by `sample_id`, `Year`, and `id` in the correct order
#       recon_long <- recon_long[order(recon_long$sample_id, recon_long$Year, recon_long$id), ]
#       
#       batch_results[[selection]][[j]] <- recon_long
#     }
#   }
#   
#   # Save batch results to an RDS file
#   saveRDS(batch_results, file = paste0(output_file, batch_number, ".rds"))
#   return(batch_results)
# }
# 
# # Split input_dfs into batches 
# batch_size <- 1
# input_ids <- names(input_dfs)
# n_batches <- ceiling(length(input_ids) / batch_size)
# 
# output_file <- "C:/Users/ib400/Github/FTT_StandAlone/Emulation/data/predictions/set_batch_results_"
# 
# for (i in 1:n_batches) {
#   batch <- input_ids[((i - 1) * batch_size + 1):min(i * batch_size, length(input_ids))]
#   process_batch(batch, i, output_file)
#   print(paste("batch", i, "processed"))
# }
# 
# 
# 
# save_batches <- function(n_batches, group_size = 1) {
# 
#   # Divide batches into groups
#   batch_groups <- split(1:n_batches, ceiling(seq_along(1:n_batches) / group_size))
#   
#   for (group_idx in c(1:n_batches)) {
#     group <- batch_groups[[group_idx]]
#     group_reconstructions <- list()
#     
#     for (i in group) {
#       # Load batch results
#       batch_results <- readRDS(paste0("C:/Users/ib400/Github/FTT_StandAlone/Emulation/data/predictions/set_batch_results_", i, ".rds"))
#       
#       # Process each emulator in the batch
#       for (emulator in names(batch_results)) {
#         df <- data.frame(batch_results[[emulator]][[1]])
#         unique_ids <- unique(df$id)
#         
#         # Process each `id`
#         for (id in unique_ids) {
#           input_data <- input_dfs[[id]]
#           unique_sample_ids <- unique(df$sample_id[df$id == id])
#           
#           expanded_data <- do.call(rbind, lapply(unique_sample_ids, function(sample_id) {
#             sample_data <- input_data[sample_id, , drop = FALSE]
#             years <- unique(df$Year[df$id == id & df$sample_id == sample_id])
#             expanded_sample <- sample_data[rep(1, length(years)), ]
#             expanded_sample$Year <- years
#             expanded_sample$sample_id <- sample_id
#             expanded_sample$id <- id
#             return(expanded_sample)
#           }))
#           
#           df <- merge(df, expanded_data, by = c("id", "sample_id", "Year"), all.x = TRUE)
#         }
#         
#         # Store processed data for the current emulator
#         group_reconstructions[[length(group_reconstructions) + 1]] <- df
#       }
#     }
#     
#     # Combine group results and save
#     group_combined <- bind_rows(group_reconstructions)
#     saveRDS(group_combined, paste0("C:/Users/ib400/Github/FTT_StandAlone/Emulation/data/predictions/set_group_results_", group_idx, ".rds"))
#     print(paste("Group", group_idx, "processed and saved"))
#   }
# }
# 
# 
# 
# combine_batches <- function() {
#   # Combine all group results
#   all_group_files <- list.files(
#     path = "C:/Users/ib400/Github/FTT_StandAlone/Emulation/data/predictions",
#     pattern = "set_group_results_.*\\.rds",
#     full.names = TRUE
#   )  
#   
#   # Initialize an empty dataframe
#   combined_data <- NULL
#   
#   # Combine incrementally
#   for (file in all_group_files) {
#     # Read the current file
#     group_data <- readRDS(file)
#     
#     ## Add in conditions, comment out for all obs
#     # group_data <- group_data %>% subset(Year == 2030 &
#     #                                       emulator != 'MEWE_IN') %>%
#     #   group_by(Year, id, sample_id) %>%  # Group by relevant variables
#     #   mutate(total_value = sum(Value, na.rm = TRUE)) %>%  # Sum 'Value'
#     #   ungroup() %>%
#     #   distinct(Year, id, sample_id, total_value, .keep_all = TRUE)  # Keep one row per group
#     
#     
#     # Combine with the existing dataframe
#     if (is.null(combined_data)) {
#       combined_data <- group_data
#     } else {
#       combined_data <- bind_rows(combined_data, group_data)
#     }
#     
#     # Optional: Print progress
#     print(paste("Processed file:", file))
#     
#   }
#   return(combined_data)
# }
# 
# 
# save_batches(n_batches)
# set_plot_data <- combine_batches()

# # Save to df
# write.csv(final_results_df, 
#           file = "C:/Users/ib400/Github/FTT_StandAlone/Emulation/data/predictions/IN_polcomp.csv",
#           row.names = F)
