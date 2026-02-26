
# Load libraries ----------------------------------------------------------

library(ggplot2)
library(dplyr)
library(emtools)


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

# Global
gbl_sa_filename <- "gbl_sa.csv"

gbl_plot_title <- "sa_gbl.png"
gbl_plot_path <- paste0(data_path, "/figures/", gbl_plot_title)

# India
in_sa_filename <- "in_sa.csv"

in_plot_title <- "sa_in.png"
in_plot_path <- paste0(data_path, "/figures/", in_plot_title)



# Load input data ---------------------------------------------------------

input_df_rescaled <- read.csv(input_path_rescaled)
inputs <- input_df_rescaled[,2:length(names(input_df_rescaled))]

# Indentify inputs to constrain e.g. cannibalisation rates
inputs_red <- c()
# set the bounds somewhere between 0-1
inputs_red_bounds <- list(min = 0.45, max = 0.55)


# Set up functions --------------------------------------------------------

# Function for plotting
rotate <- function(x) t(apply(x, 2, rev))

# A clean OAT‐sens function
oat_sens_pretrained <- function(ref, emulator, n = 30, lower = 0, upper = 1) {
  # ref: named numeric vector length d, e.g. rep(0.5, d)
  # emulator: what you pass to PredictGasp()
  # n: grid points per parameter
  d <- length(ref)
  varnames <- names(ref)
  sens <- numeric(d)
  names(sens) <- varnames
  
  # for each parameter i
  for(i in seq_len(d)) {
    # make an n×d matrix all = ref
    X_oat <- matrix(ref, nrow = n, ncol = d, byrow = TRUE)
    
    # choose bounds for this variable
    if (varnames[i] %in% inputs_red) {
      low_i  <- inputs_red_bounds$min
      high_i <- inputs_red_bounds$max
    } else {
      low_i  <- lower
      high_i <- upper
    }
  
    # replace column i by its own grid
    X_oat[, i] <- seq(low_i, high_i, length.out = n)
    colnames(X_oat) <- varnames
    
    # predict: PredictGasp often returns a vector or a list with $mean
    pred <- PredictGasp(as.data.frame(X_oat), emulator)
    m <- pred$mean
    
    # compute variance
    sens[i] <- var(m)
  }
  sens
}

# oat SA plot function
oaatSensvarSummaryPlot <- function(oat_sens_mat,
                                   threshold = 0.01) {
  
  # normalize & average
  normsens <- normalize(t(oat_sens_mat))  # rows: params, cols: outputs
  normsens_mean <- rowMeans(normsens)
  
  pol_suffix <- c("subsidies", "carbon prices", "phaseouts")

  # drop _pol parameters based on normalized sensitivity
  drop_pol <- grepl(paste(pol_suffix, collapse='|'), names(normsens_mean)) &
    (normsens_mean <= threshold)
  
  # filter oat_sens_mat and normsens to keep only relevant params
  if (any(drop_pol)) {
    oat_sens_mat <- oat_sens_mat[, !drop_pol, drop = FALSE]
    normsens <- normsens[!drop_pol, , drop = FALSE]
    normsens_mean <- normsens_mean[!drop_pol]
  }
  
  
  # 1) now extract names from the filtered matrix
  ynames <- rownames(oat_sens_mat)    # outputs
  xnames <- colnames(oat_sens_mat)    # parameters
  
  # 2) normalize & average
  normsens      <- normalize(t(oat_sens_mat))
  normsens_mean <- rowMeans(normsens)
  
  # 3) sort by importance
  sort_ix <- sort(normsens_mean, decreasing = TRUE, index.return = TRUE)$ix
  
  # 4) two‐panel layout with bigger left margin
  par(mar = c(15, 20, 5, 1), mfrow = c(1,2))
  layout(matrix(c(1,1,2), ncol = 3, nrow = 1))
  
  # 5) LEFT: heatmap
  image(rotate(normsens[sort_ix, ]), axes = FALSE, col = colorRampPalette(c("white", "darkgreen"))(100))
  axis(1,
       at     = seq(0,1,length.out = length(ynames)),
       labels = ynames,
       las    = 3, cex.axis = 1.6) # HERE
  axis(2,
       at     = seq(1,0,length.out = length(xnames)),
       labels = xnames[sort_ix],
       las    = 1, cex.axis = 1.6) #1.4
  # mtext('One-at-a-time sensitivity', side = 3, adj = 0, line = 2, cex = 1)
  
  # 6) RIGHT: dot‐plot
  lab_ix <- seq_along(xnames) - 0.5
  par(yaxs = 'i', mar = c(15,1,5,5))
  plot(rev(normsens_mean[sort_ix]), lab_ix,
       xlab = 'mean oaat variance (normalized)', ylab = '',
       ylim = c(0,length(xnames)), type = 'n', yaxt = 'n',
       cex.lab = 1.5,
       cex.axis = 1.5)
  abline(h = lab_ix, col = 'grey', lty = 'dashed')
  points(rev(normsens_mean[sort_ix]), lab_ix,
         col = "darkgreen", pch = 19, cex = 1.5)
  image.plot(legend.only = TRUE,
             zlim = c(0,1),
             col = colorRampPalette(c("white", "darkgreen"))(100),
             horizontal = TRUE,
             legend.args = list(text = 'Relative sensitivity',
                                side = 3, line = 1, cex = 1), 
             cex.axis = 3)
}



# Set up params and files for SA ------------------------------------------

# Reference grid, policies at 0 and techecon params at mean (0.5)
# alter these fixed values to perform SA elsewhere in input space
pol_fixed_value <- 0
techeco_fixed_value <- 0.5
ref_pol <- setNames(rep(pol_fixed_value, 15), colnames(inputs[1:15])) 
ref_techeco <- setNames(rep(techeco_fixed_value, 14), colnames(inputs[16:29]))
ref <- c(ref_pol, ref_techeco)

# List all your .rds files
rds_files_all <- list.files(emul_path, pattern = "\\.rds$", full.names = TRUE)

# SA on GBL emissions & capacites
global_param_files <- rds_files_all[c(c(1:3), c(19:24))]
global_param_files <- rds_files_all[grepl("GBL", rds_files_all)]


# SA on for India - performed in different subsection of input space
ref_pol['IN_phase_pol'] <- 0.5
ref_pol['IN_price_pol'] <- 0.5
ref_pol['IN_cp_pol'] <- 0.5

india_param_files <- rds_files_all[grepl("IN", rds_files_all)]



# Global sensitivity analysis ---------------------------------------------


# Read them into a named list
global_params_list <- setNames(
  lapply(global_param_files, readRDS),
  tools::file_path_sans_ext(basename(global_param_files))
)

# Run OAT sensitivity on each emulator
global_sens_list <- lapply(global_params_list, function(em) {
  oat_sens_pretrained(ref, em, n = 30)
})

# Stack into a matrix - THIS NEEDS EDITING IF VARS ARE EDITED
global_sens_mat <- do.call(rbind, global_sens_list)
rownames(global_sens_mat) <- c("Emissions 2030", "Emissions 2040", "Emissions 2050",
                               "Onshore 2030", "Onshore 2040", "Onshore 2050",
                                "Solar 2030", "Solar 2040", "Solar 2050")
colnames(global_sens_mat) <- c("US phaseouts", "US subsidies", "US carbon prices",
                               "CN phaseouts", "CN subsidies", "CN carbon prices",
                               "IN phaseouts", "IN subsidies", "IN carbon prices",
                               "RGS phaseouts", "RGS subsidies", "RGS carbon prices",
                               "RGN phaseouts", "RGN subsidies", "RGN carbon prices",
                               "learning rate: solar", "learning rate: wind",
                               "lifetime: solar", "lifetime: wind", 
                               "lead time: onshore", "lead time: solar",
                               "grid connection time", 
                               "cannibalisation: wind", "cannibalisation: solar",
                               "discount rate", "electricity demand", "gas prices (ccgt)", 
                               "coal prices", "technical potential")
# Save sensitivity matrix
write.csv(global_sens_mat,
          file = paste0(output_dir, gbl_sa_filename),
          row.names = F)

# Plot and save - FIG 2
png(gbl_plot_path, width = 1400, height = 1100, res = 150)
oaatSensvarSummaryPlot(global_sens_mat, threshold = 0.01)
dev.off()


# India sensitivity analysis ----------------------------------------------


# Read them into a named list
india_param_list <- setNames(
  lapply(india_param_files, readRDS),
  tools::file_path_sans_ext(basename(india_param_files))
)

# Run OAT sensitivity on each emulator
india_param_list <- lapply(india_param_list, function(em) {
  oat_sens_pretrained(ref, em, n = 30)
})

# Stack into a matrix - THIS NEEDS EDITING IF VARS ARE EDITED
india_sens_mat <- do.call(rbind, india_param_list)
rownames(india_sens_mat) <- c("Emissions 2030", "Emissions 2040", "Emissions 2050",
                                 "Onshore 2030",  "Onshore 2040", "Onshore 2050", 
                                 "Solar 2030", "Solar 2040", "Solar 2050",
                                 "Electricity Price 2030", "Electricity Price 2040", "Electricity Price 2050",
                                  "Renew. Share 2030", "Renew. Share 2040", "Renew. Share 2050")
                                
                                # "Onshore Generation 2030", "Onshore Generation 2040", "Onshore Generation 2050",
                                # "Solar PV Generation 2030", "Solar PV Generation 2040", "Solar PV Generation 2050",
colnames(india_sens_mat) <- c("US phaseouts", "US subsidies", "US carbon prices",
                               "CN phaseouts", "CN subsidies", "CN carbon prices",
                               "IN phaseouts", "IN subsidies", "IN carbon prices",
                               "RGS phaseouts", "RGS subsidies", "RGS carbon prices",
                               "RGN phaseouts", "RGN subsidies", "RGN carbon prices",
                               "learning rate: solar", "learning rate: wind",
                               "lifetime: solar", "lifetime: wind", 
                               "lead time: onshore", "lead time: solar",
                               "grid connection time", 
                               "cannibalisation: wind", "cannibalisation: solar",
                               "discount rate", "electricity demand", "gas prices (ccgt)", 
                               "coal prices", "technical potential")

# Save sensitivity matrix
write.csv(india_sens_mat,
          file = paste0(output_dir, in_sa_filename),
          row.names = F)


# Plot and save - FIG 3
png(in_plot_path, width = 1400, height = 1100, res = 150)
oaatSensvarSummaryPlot(india_sens_mat, threshold = 0.01)
dev.off()






