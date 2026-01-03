
# Code adapted from Mcneall 2023 - Sensitivity analysis

library(ggplot2)
library(dplyr)
library(emtools)



## Retrieve functions taken from other repositories
## Functons and scripts copied from ExeterUQ https://github.com/BayesExeter/ExeterUQ &
## https://github.com/JSalter90/UQ & https://github.com/MetOffice/jules_ppe_gmd
setwd('C:/Users/ib400/GitHub/FTT_StandAlone/Emulation/code/emulation_code/imported_code')
source('C:/Users/ib400/GitHub/FTT_StandAlone/Emulation/code/emulation_code/imported_code/UQ/Gasp.R') ### Created by James Salter



########################################

## Stylised workflow

########################################

rotate <- function(x) t(apply(x, 2, rev))


# # First step
# sensvar = function(oaat_pred, n, d){
#   # oaat_pred$mean is a long vector of predictions:
#   #   for parameter 1 we have n predictions,
#   #   then for parameter 2 another n, etc., total length = n*d
#   out = numeric(d)
#   for(i in 1:d){
#     idx = ((i-1)*n + 1):(i*n)
#     out[i] = var(pred$mean[idx])
#   }
#   out
# }





input_df_rescaled <- read.csv(paste0("C:/Users/ib400/Github/FTT_StandAlone/Emulation/data/scenarios/S3_scen_levels_rescaled.csv"))
inputs <- input_df_rescaled[,2:length(names(input_df_rescaled))]



# 1) A clean OAT‐sens function
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
    # replace column i by its own grid
    X_oat[,i] <- seq(lower, upper, length.out = n)
    colnames(X_oat) <- varnames
    
    # predict: PredictGasp often returns a vector or a list with $mean
    pred <- PredictGasp(as.data.frame(X_oat), emulator)
    m <- pred$mean
    
    # compute variance
    sens[i] <- var(m)
  }
  sens
}

##################################

##################################


oaatSensvarSummaryPlot <- function(oat_sens_mat,
                                   threshold = 0.01) {
  
  # # 0) drop _pol parameters with (near) zero mean sensitivity
  # param_means <- colMeans(oat_sens_mat)
  # drop_pol   <- grepl("_pol$", names(param_means)) &
  #   (param_means <= threshold)
  # if(any(drop_pol)) {
  #   oat_sens_mat <- oat_sens_mat[, !drop_pol, drop = FALSE]
  # }
  
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



###############################################################

# Plots

###############################################################

# Reference grid, policies at 0 and techecon params at mean (0.5)
#ref_pol <- setNames(rep(0, 18), colnames(inputs[1:18]))
# Just for pol values - take any inputs df

ref_pol <- setNames(rep(0, 15), colnames(inputs[1:15])) # EA, CN & US levels
ref_techeco <- setNames(rep(0.5, 15), colnames(inputs[16:30]))
ref <- c(ref_pol, ref_techeco)

# 1) List all your .rds files
rds_dir   <- "C:\\Users\\ib400\\Github\\FTT_StandAlone\\Emulation\\data\\emulators"
rds_files_all <- list.files(rds_dir, pattern = "\\.rds$", full.names = TRUE)
rds_files_all

############################################

#####  Global Parameters

############################################

#### 2030 global paramters
#  COULD WE DO THIS WITH JUST 2030 & 40??

# SA on GBL emissions & capacites
global_param_files <- rds_files_all[c(c(1:3), c(25:30))]
#global_param_files <- rds_files_all[c(1:3)]

# 2) Read them into a named list
global_params_list <- setNames(
  lapply(global_param_files, readRDS),
  tools::file_path_sans_ext(basename(global_param_files))
)

# 3) Run OAT sensitivity on each emulator
global_sens_list <- lapply(global_params_list, function(em) {
  oat_sens_pretrained(ref, em, n = 30)
})

# 4) Stack into a matrix
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
                               "lead time: offshore", "lead time: onshore", "lead time: solar",
                               "lead time: ccgt", "lead time: coal", "grid connection time",
                               "discount rate", "electricity demand", "gas prices (ccgt)", 
                               "coal prices", "technical potential")

write.csv(global_sens_mat, 
          file = "C:/Users/ib400/Github/FTT_StandAlone/Emulation/data/predictions/gbl_sa_lead_fix.csv",
          row.names = F)

# 5) (Optional) Plot them all at once
oaatSensvarSummaryPlot(global_sens_mat, threshold = 0.01)



############################################

##### India instrument comparison

############################################

#### 2030 

IN_polcompare_files <- rds_files_all[c(c(4:6), c(10:21))]

# 2) Read them into a named list
IN_polcompare_list <- setNames(
  lapply(IN_polcompare_files, readRDS),
  tools::file_path_sans_ext(basename(IN_polcompare_files))
)

# 3) Run OAT sensitivity on each emulator
IN_polcompare_list <- lapply(IN_polcompare_list, function(em) {
  oat_sens_pretrained(ref, em, n = 30)
})

# 4) Stack into a matrix
IN_polcompare_mat <- do.call(rbind, IN_polcompare_list)
rownames(IN_polcompare_mat) <- c("Emissions 2030", "Emissions 2040", "Emissions 2050",
                                 "Onshore 2030",  "Onshore 2040", "Onshore 2050", 
                                 "Solar 2030", "Solar 2040", "Solar 2050",
                                 "Electricity Price 2030", "Electricity Price 2040", "Electricity Price 2050",
                                  "Renew. Share 2030", "Renew. Share 2040", "Renew. Share 2050")
                                
                                # "Onshore Generation 2030", "Onshore Generation 2040", "Onshore Generation 2050",
                                # "Solar PV Generation 2030", "Solar PV Generation 2040", "Solar PV Generation 2050",
colnames(IN_polcompare_mat) <- c("US phaseouts", "US subsidies", "US carbon prices",
                               "CN phaseouts", "CN subsidies", "CN carbon prices",
                               "IN phaseouts", "IN subsidies", "IN carbon prices",
                               "RGS phaseouts", "RGS subsidies", "RGS carbon prices",
                               "RGN phaseouts", "RGN subsidies", "RGN carbon prices",
                               "learning rate: solar", "learning rate: wind",
                               "lifetime: solar", "lifetime: wind", 
                               "lead time: offshore", "lead time: onshore", "lead time: solar",
                               "lead time: ccgt", "lead time: coal", "grid connection time",
                               "discount rate", "electricity demand", "gas prices (ccgt)", 
                               "coal prices", "technical potential")


# 5) (Optional) Plot them all at once
oaatSensvarSummaryPlot(IN_polcompare_mat, threshold = 0.001)











