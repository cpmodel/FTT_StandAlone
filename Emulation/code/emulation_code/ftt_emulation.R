
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

setwd('C:/Users/ib400/OneDrive - University of Exeter/Desktop/PhD/GitHub/UQ')
source('code/Gasp.R')
setwd('C:/Users/ib400/OneDrive - University of Exeter/Desktop/PhD/GitHub/UQ')

#########################

## James data loading

#########################

# design <- readRDS('applications/SECRET/data/designW1.rds')
# 
# t <- 512 # number of timepoints
# v <- 3 # number of variables
# n <- 100 # number of simulations
# 
# all_data <- array(0, dim = c(t,v,n))
# for (i in 1:n){
#   tmp <- readMat(paste0('applications/SECRET/data/output/flow', i, '.mat'))[[1]]
#   all_data[,,i] <- tmp
# }
# 
# # Just for visualising the data
# check_j_data <- data.frame(tmp)
# 
# ## James plotting
# plot_data <- data.frame(Time = 1:t,
#                         Run = rep(1:n, each = t*v),
#                         Output = c(all_data),
#                         Type = rep(c('Flow1', 'Flow2', 'Pressure'), each = t))
# 
# plot_data2 <- melt(plot_data, id.vars = c('Time', 'Run', 'Type'))
# 
# 
# ggplot(plot_data2, aes(x = Time, y = value, col = as.factor(Run))) +
#   geom_line() +
#   facet_wrap(vars(Type), nrow = 2, scales = 'free_y') +
#   theme(legend.position = 'none')


#########################

####  Ian's data loading

#########################
# 

# This needs specifying for later, 
t <- 41 # time (issues as different number of time points in ling format data)
v <- 1 # variables
n <- 200 # no. simulations 


### Attempt to load data in the same way as James but not working 
# all_data_ftt <- array(0, dim = c(t,v,n))
# 
# for (i in 0:n){
#   tmp <- read.csv(paste0('../FTT_StandAlone/Emulation/data/batch_', i, '.csv')) # column with output data?
#   all_data_ftt[,,i] <- tmp
# }

##### OUTPUT MATRIX

csv_files <- list.files(path = "../FTT_StandAlone/Emulation/data", 
                        pattern = "batch_.*\\.csv", full.names = T)

csv_files <- mixedsort(csv_files)
# Read each CSV file and store them in a list
dfs <- lapply(csv_files, read_csv)
combined_df <- bind_rows(dfs)
combined_df <- combined_df %>% subset(scenario != 'S3_200')

# Select variable - global cumul capacity to start
meww_df_sol <- subset(combined_df, variable == 'MEWW' & technology == '19 Solar PV')
meww_df_onshore <- subset(combined_df, variable == 'MEWW' & technology == '17 Onshore')
meww_df_offshore <- subset(combined_df, variable == 'MEWW' & technology == '18 Offshore')

## German prices
metc_sol_de <- subset(combined_df, variable == 'METC' & technology == '19 Solar PV' & country_short == 'DE')
metc_onshore_de <- subset(combined_df, variable == 'METC' & technology == '17 Onshore' & country_short == 'DE')
metc_offshore_de <- subset(combined_df, variable == 'METC' & technology == '18 Offshore'& country_short == 'DE')

## German capacity
mewk_sol_de <- subset(combined_df, variable == 'MEWK' & technology == '19 Solar PV' & country_short == 'DE')
mewk_sol_de <- subset(combined_df, variable == 'MEWK' & technology == '17 Onshore' & country_short == 'DE')
mewk_sol_de <- subset(combined_df, variable == 'MEWK' & technology == '18 Offshore' & country_short == 'DE')

# Delete to save memory
dfs <- NULL
#combined_df <- NULL
####### Plot output for single output

# skip for full output

#ggplot(meww_df, aes(x = year, y = value, col = as.factor(scenario))) +
  # geom_line() +
  # facet_wrap(vars(technology), nrow = 2, scales = 'free_y') +
  # theme(legend.position = 'none') +
  # xlab("Year") + ylab("Global Culmulative Capacity (GW")


#scale values in 'sales' column to be between 0 and 1
#meww_2050_sol <- subset(meww_df, year == 2050 & technology == '19 Solar PV')


#### DESIGN MATRIX

### Load design matrix (inputs)
design_ftt <- read.csv("../FTT_StandAlone/Emulation/scenarios/S3_scenario_levels.csv")

# change id col for merge
colnames(design_ftt)[1] <- "scenario"
# This scenario causing issues
design_ftt <- design_ftt %>% subset(scenario != 'S3_200')


# Rescaling of values between 0-1, only based on values in the scenarios
#### NEEDS CHANGING
design_ftt$learning_solar <- rescale(design_ftt$learning_solar)
design_ftt$learning_wind <- rescale(design_ftt$learning_wind)
design_ftt$lifetime_solar <- rescale(design_ftt$lifetime_solar)
design_ftt$lifetime_wind <- rescale(design_ftt$lifetime_wind)
design_ftt$grid_expansion_lead <- rescale(design_ftt$grid_expansion_lead)
design_ftt$south_discr <- rescale(design_ftt$south_discr)
design_ftt$north_discr <- rescale(design_ftt$north_discr)



# #Merge design and output for single output
# tData <- merge(design_ftt, meww_2050_sol[,c("scenario", "y")], by = "scenario", all = FALSE)
# 
# em_data <- data.frame(tData[,2:(ncol(tData)-1)],
#                       Noise = runif(nrow(tData), -1,1),
#                       y = tData$y)



# ## Build first emulator with seed for training set
# set.seed(2101)
# em1 <- BuildGasp('y', em_data)
# summary(em1)
# 
# ## Validation
# ## Predicting over validation data
# par(mar = c(4,2,2,2));ValidateGasp(em1)
# 
# # can also provide with new dataset
# par(mfrow = c(2,3), mar = c(4,2,2,2));ValidateGasp(em1, IndivPars = TRUE)
# 
# # leave-one-out across the training data:
# par(mar = c(4,2,2,2));LeaveOneOut(em1)
# 
# # Adding mean function
# set.seed(5820)
# em2 <- BuildGasp('y', em_data, mean_fn = 'linear')
# 
# # More general mean function
# set.seed(3100329)
# em3 <- BuildGasp('y', em_data, mean_fn = 'step')
# 
# # Validate
# par(mfrow=c(1,2),mar = c(4,2,2,2));ValidateGasp(em3);LeaveOneOut(em3)
# 
# summary(em3)
# 
# 
# summary(em3$lm$linModel)
# em3$active

######################################
## Emulate full output
######################################

# dimension reduction

##############

# Jame's - create basis

# ##############
# DataBasis <- MakeDataBasis(all_data[,1,])
# summary(DataBasis)
# dim(DataBasis$tBasis)
# dim(DataBasis$CentredField)

##############

# Ian's - result is scenarios are columns, time as rows

##############
# # Convert the dataframe to wide format with 'scenario' as columns, 'year' as rows and 'value'
# Solar glob cap
meww_data_sol <- dcast(meww_df_sol, year ~ scenario, value.var = "value")
meww_data_sol$year <- NULL # don't need the year col now
# Reordering of model runs (columns)
meww_data_sol <- meww_data_sol[,match(design_ftt$scenario,colnames(meww_data_sol))]

# Onshore glob cap
meww_data_onshore <- dcast(meww_df_onshore, year ~ scenario, value.var = "value")
meww_data_onshore$year <- NULL # don't need the year col now
# Reordering of model runs (columns)
meww_data_onshore <- meww_data_onshore[,match(design_ftt$scenario,colnames(meww_data_onshore))]
 
# Offshore glob cap
meww_data_offshore <- dcast(meww_df_offshore, year ~ scenario, value.var = "value")
meww_data_offshore$year <- NULL # don't need the year col now
# Reordering of model runs (columns)
meww_data_offshore <- meww_data_offshore[,match(design_ftt$scenario,colnames(meww_data_offshore))]

# German prices
metc_data_sol_de <- dcast(metc_sol_de, year ~ scenario, value.var = "value")
metc_data_sol_de$year <- NULL # don't need the year col now
# Reordering of model runs (columns)
metc_data_sol_de <- metc_data_sol_de[,match(design_ftt$scenario,colnames(metc_data_sol_de))]

# German capacity
mewk_data_sol_de <- dcast(mewk_sol_de, year ~ scenario, value.var = "value")
mewk_data_sol_de$year <- NULL # don't need the year col now
# Reordering of model runs (columns)
mewk_data_sol_de <- mewk_data_sol_de[,match(design_ftt$scenario,colnames(mewk_data_sol_de))]

# German prices
metc_data_sol_de <- dcast(metc_sol_de, year ~ scenario, value.var = "value")
metc_data_sol_de$year <- NULL # don't need the year col now
# Reordering of model runs (columns)
metc_data_sol_de <- metc_data_sol_de[,match(design_ftt$scenario,colnames(metc_data_sol_de))]


##############################################################

######### Make basis

##############################################################


# Change variable here for different variables x tech
output <- meww_data_sol

DataBasis_ftt <- MakeDataBasis(as.matrix(output))
summary(DataBasis_ftt)
dim(DataBasis_ftt$tBasis)
dim(DataBasis_ftt$CentredField)

# design mat for emulating full output
# seems to be in correct order even for subset of batches
tData_ftt <- design_ftt


# Number of basis (q)
q <- 8
t <- 41
plot_basis <- data.frame(Time = rep(1:t, q),
                         Vector = rep(1:q, each = t),
                         Weight = c(DataBasis_ftt$tBasis[,1:q]))
ggplot(plot_basis, aes(Time, Weight)) +
  geom_line() +
  facet_wrap(vars(Vector))

# How many vectors explain this percentage of the variablility
q <- ExplainT(DataBasis_ftt, vtot = 0.9999)
q
q <- 4

# This depends on 'n' defined at the start, it is the length of coeffs not no. sims
vars <- lapply(1:ncol(DataBasis_ftt$tBasis), function(k) VarExplained(DataBasis_ftt$tBasis[,1:k], DataBasis_ftt$CentredField))
# plot the basis
ggplot(data.frame(q = 1:ncol(DataBasis_ftt$tBasis), Proportion = unlist(vars)), aes(x = q, y = Proportion)) +
  geom_line() +
  ylim(0,1)

Coeffs <- Project(data = DataBasis_ftt$CentredField, 
                  basis = DataBasis_ftt$tBasis[,1:q])
colnames(Coeffs)[1:q] <- paste("C",1:q,sep="")
summary(Coeffs)

# combine coefficients with design and add noise vector required
tDataC <- data.frame(tData_ftt[,-1], # getting the scaled version from before
                     Noise = runif(nrow(tData_ftt), -1, 1), 
                     Coeffs)
head(tDataC)

set.seed(321)
inds <- sample(1:nrow(tData_ftt), nrow(tData_ftt))
# Set training and test data
train_inds <- inds[1:180]
val_inds <- inds[-c(1:180)]
train_data <- tDataC[train_inds,]
val_data <- tDataC[val_inds,]
 
# Emulators all entries
em_coeffs <- BasisEmulators(tDataC, q, mean_fn = 'step', maxdf = 5, training_prop = 1)

# Emulators split into training testing
#em_coeffs <- BasisEmulators(train_data, q, mean_fn = 'step', maxdf = 5, training_prop = 1)

### Summary and info on coeffs, how can this be interpreted?
summary(em_coeffs[[1]])
summary(em_coeffs[[1]]$lm$linModel)
summary(em_coeffs[[2]])
summary(em_coeffs[[2]]$lm$linModel)
summary(em_coeffs[[3]])
summary(em_coeffs[[3]]$lm$linModel)


#Now in `ValidateGasp`, we need to provide the validation set (it's no longer internal to the `BasisEmulator` object):


par(mfrow=c(2,2), mar=c(4,2,2,2))
ValidateGasp(em_coeffs[[1]], val_data)
ValidateGasp(em_coeffs[[2]], val_data)
ValidateGasp(em_coeffs[[3]], val_data)

par(mfrow=c(2,2), mar=c(4,2,2,2))
LeaveOneOut(em_coeffs[[1]]);LeaveOneOut(em_coeffs[[2]]);LeaveOneOut(em_coeffs[[3]]); LeaveOneOut(em_coeffs[[4]])



## Prediction

#The validation plots above are performing prediction across the test data. 
#In general, we can predict for any sets of inputs, for either a 1D emulator,
#or for a set of basis emulators. Doing so across a space-filling design in parameter space 
# (here, we are still working in [-1,1]):


ns <- 1000 # usually want more, but set low for speed
vars <- 17 # how many inputs
BigDesign <- as.data.frame(randomLHS(ns, vars))
colnames(BigDesign) <- colnames(tData_ftt)[2:18] # input cols without scen
#Preds_1D <- PredictGasp(BigDesign, em3)
Preds_basis <- BasisPredGasp(BigDesign, em_coeffs) # produces basis

#These store slightly different things - in the 1D case, we get mean/sd/lower95/upper95 (the same as what `predict.rgasp` returns):
#summary(Preds_1D)
summary(Preds_basis)

#In the basis case, I only store `$Expectation` and `$Variance` (as this is all we need for history matching, and because prediction intervals can be derived from these, so don't want to store these if we have a lot of emulators):
dim(Preds_basis$Expectation)
dim(Preds_basis$Variance)


## Reconstructing
#From basis coefficients (whether given by projection, or predicted by an emulator), we can reconstruct a prediction of the original field.

# Produce prediction for the validation data
Preds_val <- BasisPredGasp(val_data, em_coeffs)
#Let's consider the runs from the validation set. First produce predictions for them:
# i is the number of the validation case
i <- 1
# With ensemble mean which dominates
plot_data <- data.frame(Year = 2010:2050,
                        Truth = DataBasis_ftt$EnsembleMean + DataBasis_ftt$CentredField[,val_inds[i]],
                        Recon = DataBasis_ftt$EnsembleMean + Recon(Preds_val$Expectation[i,], DataBasis_ftt$tBasis[,1:q]))
# Just wrangle to plot
plot_data2 <- melt(plot_data, id.vars = c('Year'))
ggplot(plot_data2, aes(Year, value, col = variable)) + geom_line()

# Without ensemble mean
plot_data <- data.frame(Year = 2010:2050,
                        Truth = DataBasis_ftt$CentredField[,val_inds[i]],
                        Recon = Recon(Preds_val$Expectation[i,], DataBasis_ftt$tBasis[,1:q]))
plot_data2 <- melt(plot_data, id.vars = c('Year'))



ggplot(plot_data2, aes(Year, value, col = variable)) + geom_line()


# Mean prediction by itself is not that informative, also sample from the emulators (i):
i <- 1
em_samp <- matrix(0, t, 1000) # last number is size of sample
for (s in 1:1000){
  samp <- rnorm(q, 
                mean = Preds_val$Expectation[i,], # i is validation point
                sd = sqrt(Preds_val$Variance[i,]))
  rec <- Recon(samp, DataBasis_ftt$tBasis[,1:q])
  em_samp[,s] <-  rec
}

check <- data.frame(em_samp)

## Plot truth vs reconstruction
ggplot(plot_data2, aes(Year, value, col = variable)) + 
  geom_line(data = data.frame(Year = 2010:2050, value = c(em_samp), s = rep(1:1000, each = t)), aes(Year, value, linetype = as.factor(s)), col = 'light blue', alpha = 0.6) +
  geom_line(size = 1.25) +
  scale_linetype_manual(values = rep(1,1000), guide = 'none')

#### Representing uncertainty due to basis vectors left out

########## Start back here

###########################

extra_var <- DiscardedBasisVariance(DataBasis_ftt, q)

## With my data this is currently not a pos def matrix, adding a small value to the
## diagonal temporarily to allow for plotting
# Add a small positive value to the diagonal elements
small_value <- 1e-10 # Adjust as needed
diag(extra_var) <- diag(extra_var) + small_value


## Sampling not working, just taking portion of variance mat for now
extra_var_samples <- rmvnorm(1000, rep(0, t), extra_var)


# Need to inspect the dimensions to know which numbers to put in segment below
dim(extra_var_samples)

plot_samples <- data.frame(Year = 2010:2050,
                           epsilon = c(t(extra_var_samples)),
                           s = rep(1:1000, each = t))

# Samples from the discarded vectors and there is some correlated structure here,
# this is just drowned out
ggplot(plot_samples, aes(Year, epsilon, col = as.factor(s))) + 
  geom_line(alpha = 0.6) +
  theme(legend.position = 'none')

## Adding this to emulator samples
ggplot(plot_data2, aes(Year, value, col = variable)) + 
  geom_line(data = data.frame(Year = 2010:2050, value = c(em_samp + t(extra_var_samples)), 
                              s = rep(1:1000, each = 41)), aes(Year, value, linetype = as.factor(s)), 
            col = 'grey', alpha = 0.6) +
  geom_line(size = 1.25) +
  scale_linetype_manual(values = rep(1,1000), guide = 'none')


## Before, we were missing the data in places, because the truncated basis did not have the ability
## to perfectly reconstruct the truth. Now, the truth lies within our uncertainty.

#Just plotting 95% prediction intervals for clarity:
plot_data <- data.frame(Time = 1:t,
                        Truth = DataBasis_ftt$CentredField[,val_inds[1]],
                        Recon = rep(Recon(Preds_val$Expectation[1,], DataBasis_ftt$tBasis[,1:q]), 2),
                        Lower = c(apply(em_samp, 1, quantile, probs = 0.025), apply(em_samp + t(extra_var_samples), 1, quantile, probs = 0.025)),
                        Upper = c(apply(em_samp, 1, quantile, probs = 0.975), apply(em_samp + t(extra_var_samples), 1, quantile, probs = 0.975)),
                        Type = rep(c('EmVar', 'FullVar'), each = t))
plot_data2 <- melt(plot_data, id.vars = c('Time', 'Type'))
pal <- scales::hue_pal()(2)
ggplot(plot_data2, aes(Time, value, col = variable, linetype = variable)) +
  geom_line(size = 0.8) +
  facet_wrap(vars(Type)) +
  scale_linetype_manual(values = c(1,1,2,2)) +
  scale_colour_manual(values = pal[c(1,2,2,2)]) +
  theme(legend.position = 'none')

# Or doing for 4 different inputs (immediately adding in the variance from the discarded vectors):

em_samp <- array(0, dim = c(t, 100, 4))
plot_data <- NULL
for (i in 2:5){
  plot_data <- rbind(plot_data,
                     data.frame(Time = 1:t,
                                Truth = DataBasis_ftt$CentredField[,val_inds[i]],
                                Recon = Recon(Preds_val$Expectation[i,], DataBasis_ftt$tBasis[,1:q]),
                                Run = i))
  for (s in 1:100){
    samp <- rnorm(q, 
                  mean = Preds_val$Expectation[i,],
                  sd = sqrt(Preds_val$Variance[i,]))
    rec <- Recon(samp, DataBasis_ftt$tBasis[,1:q])
    em_samp[,s,i-1] <- rec
  }
}

plot_data2 <- melt(plot_data, id.vars = c('Time', 'Run'))

extra_var_samples <- t(rmvnorm(400, rep(0, t), extra_var))
dim(extra_var_samples) <- c(t, 100, 4)

plot_data_samp <- data.frame(Time = 1:t, 
                             value = c(em_samp + extra_var_samples), 
                             s = rep(1:100, each = t),
                             Run = rep(2:5, each = t*100))

custom_labels <- c(
  "1" = "Validation 1",
  "2" = "Validation 1",
  "3" = "Validation 2",
  "4" = "Validation 3",
  "5" = 'Validation 4'
  # Add more custom labels as needed
)
ggplot(plot_data2, aes(Time, value, col = variable)) +
  facet_wrap(vars(Run), scales = 'free_y', labeller = labeller(Run = custom_labels)) +
  geom_line(data = plot_data_samp, aes(Time, value, linetype = as.factor(s)), col = 'light blue', alpha = 0.6) +
  geom_line(size = 0.8) +
  scale_linetype_manual(values = rep(1, 100), guide = 'none') +
  scale_x_continuous(
    breaks = seq(0, 41, by = 10),     # Set the breaks as 0 to 41
    labels = seq(2010, 2051, by = 10) # Set the labels as 2010 to 2050
  ) +
  labs(
    x = "Year",                      # Label for x-axis
    y = "Difference from mean",             # Label for y-axis
    color = "Variables",             # Label for the color legend
    linetype = "Sample Type"         # Label for the linetype legend (if guide is used)
  )


##############

## Visualising response surface

#We may want to see what's happening across the output space is we systematically
#change values of the inputs. Here, the 2 dominant parameters are `kMV` and `alpha`, 
#so let's vary these.

#Before we predicted over a large Latin hypercube. 
#Now let's systematically vary these 2 parameters, 
#and fix the others at the centre of their range (zero):

pred_seq <- seq(from = 0, to = 1, by = 0.02)
pred_grid <- expand.grid(E._cp = pred_seq, CN_cp = pred_seq)
pred_grid <- expand.grid(E._reg = pred_seq, CN_reg = pred_seq)
ngrid <- nrow(pred_grid)
NewDesign <- data.frame(pred_grid, US_cp = 0.5, E._cp = 0.5,
                        north_discr = 0.5, IN_cp = 0.5,
                        grid_expansion_lead = 1, CN_cp = 0.5, 
                        ROW_cp = 0.5, ROW_reg = 0.5, IN_reg = 0.5,
                        learning_wind = 0.5, lifetime_solar = 0.5, lifetime_wind = 0.5,
                         learning_solar = rep(c(0.4,0.5,0.6), each = nrow(pred_grid)), 
                        south_discr = 0.5, US_reg = 0.5)

pred_grid <- expand.grid(E._cp = pred_seq, CN_cp = pred_seq)
ngrid <- nrow(pred_grid)
NewDesign <- data.frame(pred_grid, US_cp = 0.5, E._reg = 0.5,
                        north_discr = 0.5, IN_cp = 0.5,
                        grid_expansion_lead = 1, CN_reg = 0.5, 
                        ROW_cp = 0.5, ROW_reg = 0.5, IN_reg = 0.5,
                        learning_wind = 0.5, lifetime_solar = 0.5, lifetime_wind = 0.5,
                        learning_solar = rep(c(0.4,0.5,0.6), each = nrow(pred_grid)), 
                        south_discr = 0.5, US_reg = 0.5)
pred_grid <- expand.grid(E._reg = pred_seq, US_reg = pred_seq)
ngrid <- nrow(pred_grid)
NewDesign <- data.frame(pred_grid, US_cp = 0.5, E._cp = 0.5,
                        north_discr = 0.5, IN_cp = 0.5,
                        grid_expansion_lead = 1, CN_reg = 0.5, 
                        ROW_cp = 0.5, ROW_reg = 0.5, IN_reg = 0.5,
                        learning_wind = 0.5, lifetime_solar = 0.5, lifetime_wind = 0.5,
                        learning_solar = rep(c(0.4,0.5,0.6), each = nrow(pred_grid)), 
                        south_discr = 0.5, CN_cp = 0.5)

PredsNewBasis <- BasisPredGasp(NewDesign, em_coeffs)

PredsRecon <- matrix(0, nrow = nrow(DataBasis_ftt$tBasis), ncol = nrow(PredsNewBasis$Expectation)) # whatever the original dimension ell was

for (i in 1:nrow(PredsNewBasis$Expectation)) {PredsRecon[,i] <- Recon(PredsNewBasis$Expectation[i,], 
                                         DataBasis_ftt$tBasis[,1:q]) + DataBasis_ftt$EnsembleMean} # add ensemble mean if needed, otherwise delete this part
check <- data.frame(NewDesign, Global_Solar_Capacity = PredsRecon[41,])

NewDesign$learning_solar <- factor(NewDesign$learning_solar, 
                                   levels = c(0.6, 0.5, 0.4), 
                                   labels = c("Low learning", "Mid learning", "High learning"))
# #custom_labels <- c(
#   "0.6" = "Low learning rate",
#   "0.5" = "Mid learning rate",
#   "0.4" = "High learning rate"
#   # Add more custom labels as needed
# )
cont_range = seq(18500, 19500, by = 1000)
# Plotting the expectation :
ggplot(data.frame(NewDesign, Global_Solar_Capacity = PredsRecon[41,]),
       aes(x = CN_reg, y = E._reg, z = Global_Solar_Capacity,
           col = Global_Solar_Capacity)) +
  geom_point(size = 3, shape = 15) +
  geom_contour(breaks = cont_range, col = 'white') +
  scale_colour_viridis() +
  facet_wrap(~learning_solar) +
  labs(
    x = "China Regulatory Ambition",  # Label for x-axis
    y = "European Regulatory Ambition",   # Label for y-axis
    color = "Global Solar Capacity (GW in 2050)" # Label for color legend
  )
# Plotting the expectation :
ggplot(data.frame(NewDesign, Global_Solar_Capacity = PredsRecon[41,]),
       aes(x = CN_cp, y = E._cp, z = Global_Solar_Capacity,
           col = Global_Solar_Capacity)) +
  geom_point(size = 3, shape = 15) +
  geom_contour(breaks = cont_range, col = 'white') +
  scale_colour_viridis() +
  facet_wrap(~learning_solar) +
  labs(
    x = "China Tax Ambition",  # Label for x-axis
    y = "Europe Tax Ambition",   # Label for y-axis
    color = "Global Solar Capacity (GW in 2050)" # Label for color legend
  )



em_coeffs[[1]]$lm$linModel

# Or `C2`:
ggplot(data.frame(NewDesign, C2 = PredsNewBasis$Expectation[,2]),
       aes(x = kMV, y = alpha, col = C2)) +
  geom_point(size = 3, shape = 15) +
  scale_colour_viridis()


# This is a 2D surface in the 4D input space 
ggplot(data.frame(NewDesign, C1 = PredsNewBasis$Expectation[,1]),
       aes(x = kMV, y = alpha, z = C1, col = C1)) +
  geom_point(size = 3, shape = 15) +
  geom_contour(breaks = c(-5), col = 'white') +
  scale_colour_viridis(limits = c(-85,85))

