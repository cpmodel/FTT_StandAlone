# Create basis (DataBasis object) and other preliminary calculations (inverting W)
CreateBasis <- function(data, type = 'SVD', W = NULL, RemoveMean = TRUE){
  # Invert W, creating Winv and tagging with attributes to enable fast calculations later on
  if (is.null(W)){
    W <- diag(nrow(data))
  }
  
  Winv <- GetInverse(W)

  if (type %in% c('SVD', 'svd', 'L2')){
    DataBasis <- MakeDataBasis(data, weightinv = NULL, RemoveMean = RemoveMean, StoreEigen = TRUE)
  }
  
  if (type %in% c('WSVD', 'wsvd', 'wSVD')){
    DataBasis <- MakeDataBasis(data, weightinv = Winv, RemoveMean = RemoveMean, StoreEigen = TRUE)
  }
  
  DataBasis$Type <- type
  
  if (DataBasis$Type %in% c('svd', 'L2')){
    DataBasis$Type <- 'SVD'
  }
  
  if (DataBasis$Type %in% c('WSVD', 'wsvd')){
    DataBasis$Type <- 'wSVD'
  }
  
  return(DataBasis)
}





AssessBasis <- function(DataBasis, Obs){
  
  max_q <- dim(DataBasis$tBasis)[2]
  PlotData <- data.frame(Size = 1:max_q, Error = numeric(max_q), Explained = numeric(max_q))
    
  if (!is.null(DataBasis$scaling)){
    PlotData$Error <- errors(DataBasis$tBasis, Obs - DataBasis$EnsembleMean, DataBasis$Winv)*DataBasis$scaling^2
  }
  else {
    PlotData$Error <- errors(DataBasis$tBasis, Obs - DataBasis$EnsembleMean, DataBasis$Winv)
  }
  
  if (is.null(DataBasis$Winv)){
    var_sum <- crossprod(c(DataBasis$CentredField))
  }
  else {
    var_sum <- sum(diag(t(DataBasis$CentredField) %*% DataBasis$Winv %*% DataBasis$CentredField))
  }
  
  for (i in 1:max_q){
    PlotData$Explained[i] <- VarExplained(DataBasis$tBasis[,1:i], DataBasis$CentredField, DataBasis$Winv, total_sum = var_sum)
  }
  
  PlotData$Explained <- round(PlotData$Explained, 10) # to make sure plots when = 1
  
  chi_bound <- qchisq(0.995, nrow(DataBasis$tBasis)) / nrow(DataBasis$tBasis)
  max_y <- max(c(PlotData$Error, chi_bound + 0.05))

  var_plot <- ggplot(data = PlotData, aes(x = Size)) +
    geom_line(aes(y = Error), col = 'red') +
    geom_line(aes(y = Explained * max_y), col = 'blue') +
    xlab('Basis size') +
    scale_y_continuous(
      #name = 'Error',
      limits = c(0,max_y),
      #breaks = seq(from = 0, to = 1*max_y, by = 0.25*max_y),
      #labels = NULL,
      sec.axis = sec_axis(trans=~./max_y, name="Explained")) +
    #geom_vline(xintercept = q) +
    geom_hline(yintercept = chi_bound, linetype = 'dashed', col = 'red') +
    geom_hline(yintercept = 0.9*max_y, linetype = 'dashed', col = 'blue')
  #theme(panel.grid.major = element_blank(), 
  #      panel.grid.minor = element_blank())

  # Then don't need to do ExplainT, as already have the error/explained combinations
  # Give list of (threshold, q)
  thresholds <- c(0.8, 0.85, 0.9, 0.95, 0.99, 0.999)
  q <- numeric(length(thresholds))
  for (j in 1:length(q)){
    q[j] <- which(PlotData$Explained >= thresholds[j])[1]
  }
  
  return(list(plot = var_plot,
              Errors = PlotData,
              Truncations = data.frame(Threshold = thresholds,
                             q = q)))
}


# Use this to create tData, projected obs, etc.
#### POSSIBLY RENAME ####
GetEmulatableData <- function(Design, DataBasis, BasisSize = NULL, Obs, Noise = TRUE){
  if(Noise){
    Noise <- runif(length(Design[,1]),-1,1)
    Design <- cbind(Design, Noise)
  }
  tData <- Project(DataBasis$CentredField, DataBasis$tBasis[,1:BasisSize], weightinv = DataBasis$Winv)
  colnames(tData) <- paste0('C', 1:ncol(tData)) 
  tData <- cbind(Design, tData)
  return(tData)
}


MVImplausibilityMOGP <- function(NewData, Emulator, DataBasis, Discrepancy, Obs, ObsErr){
  tEmulator <- Emulator$mogp$predict(as.matrix(NewData), deriv=FALSE)
  
  HistoryMatch(DataBasis, Obs, t(tEmulator$mean), t(tEmulator$unc), ObsErr, Discrepancy, weightinv = DataBasis$Winv)
  #n <- ncol(tEmulator$mean)
  #tpreds <- t(tEmulator$mean)
  #tunc <- t(tEmulator$unc)
  #as.numeric(mclapply(1:n, function(k) ImplCoeff(tpreds[k,], tunc[k,], Obs, ObsErr, Discrepancy)))
}


BuildNewEmulatorsFourier <- function(tData, HowManyEmulators, 
                              additionalVariables=NULL, 
                              Choices = lapply(1:HowManyEmulators,
                                               function(k) choices.default), meanFun = "linear", 
                              kernel = c("Gaussian"),
                              SpecFormula = NULL,
                              Active = NULL,
                              ...){
  #'@description Builds MO_GP emulators for a data frame 
  #'@param tData This is the data frame you wish to use. The format should be D+1+Q where Q is the number of targets each occupying one of the last Q columns, D is the number of inputs occupying the first D columns. The D+1th column should be a vector of random normal numbers (between -1 and 1) called "Noise". We use it in our LM code to stop our algorithms overfitting by selecting signals by chance. All variables should have names.
  #'@param HowManyEmulators How many emulators are required. The code will fit the first HowManyEmulators GPs up to Q.
  #'@param additionalVariables Are there variables that must be "active" (e.g. you really want to use them in a decision problem or similar) or should be included? Often all variables are specified here, but it defaults to NULL.
  #'@param Choices A list of choices with each of the HowManyEmulators elements being a choice list compatible with GetPriors() (see the documentation for GetPriors)
  #'@param meanFun Currently a single string either "fitted", or "linear" ("constant" to come). If "fitted", our custom global mean functions are fitted and then used to fit the GP. Recommended for higher dimensions and for history matching. If "linear", a linear mean function is fitted to all emulators and using additionalVariables. A list implementation will be considered in future versions. Could also make a list where it could be a formula (is.formula)
  #'@param kernel A vector of strings that corresponds to the type of kernel either "Gaussian", or "Matern52"
  #'Default is to use Gaussian kernel for all emulators.
  #'@details If mean functions are not given (an option that will be added soon) our automatic LM code fits a global mean function for each metric. Get Priors is then used to extract the priors before we establish an MOGP and fit the parameters by MAP estimation. MAP improves on MLE here as we avoid the ridge on the likelihood surface for GPs.
  #'@return A list with 2 elements. 1 the mogp, 2 a list containing the elements used for fitting: the mean functions (containing element linModel as the lm object), the design, a list of active input indices for each emulator (to be used for subsetting the design to produce diagnostics), and the prior choices.
  ###Mean function selection###
  lastCand <- which(names(tData)=="Noise")
  if(length(lastCand)<1)
    stop("tData should have a column called 'Noise' separating the inputs and outputs.")
  if(is.null(HowManyEmulators))
    HowManyEmulators <- length(names(tData)) - lastCand
  if(!(HowManyEmulators == length(names(tData)) - lastCand)){
    tData <- tData[,c(1:lastCand,(lastCand+1):(lastCand+HowManyEmulators))]
  }
  if(meanFun =="fitted"){
    tdfs <- DegreesFreedomDefault(Choices, N=length(tData[,1]))
    lm.list = lapply(1:HowManyEmulators, function(k) 
      try(EMULATE.lm(Response=names(tData)[lastCand+k],
                     tData=tData, tcands=names(tData)[1:lastCand],
                     tcanfacs=NULL, 
                     TryFouriers=Choices[[k]]$lm.tryFouriers, 
                     maxOrder=Choices[[k]]$lm.maxOrder,
                     maxdf = tdfs[k])))
  }
  else if(meanFun == "linear"){
    if(is.null(additionalVariables))
      stop("When specifying linear meanFun, please pass the active inputs into additionalVariables")
    linPredictor <- paste(additionalVariables,collapse="+")
    lm.list = lapply(1:HowManyEmulators, function(k) list(linModel=eval(parse(text=paste("lm(", paste(names(tData[lastCand+k]), linPredictor, sep="~"), ", data=tData)", sep="")))))
  }
  #### add option for specified mean function ####
  else if(meanFun == 'spec'){
    lm.list = lapply(1:HowManyEmulators, function(k) list(linModel = eval(parse(text=paste("lm(", paste(names(tData[lastCand+k]), SpecFormula[k], sep="~"), ", data=tData)", sep="")))))
  }
  
  
  else{
    stop("meanFun must either be 'fitted' or 'linear' in this version")
  }
  ###Prepare the data for MOGP### 
  tfirst <- lastCand + 1
  target_names <- names(tData)[tfirst:length(names(tData))]
  target_list <- extract_targets(tData[,-which(names(tData)=="Noise")], target_names)
  inputs <- target_list[[1]]
  targets <- target_list[[2]]
  inputdict <- target_list[[3]]
  d <- dim(inputs)[2]
  
  if(meanFun=="fitted"){
    ActiveVariableIndices <- lapply(lm.list, function(tlm) which((names(tData)%in%additionalVariables) | (names(tData)%in%tlm$Names) | (names(tData) %in% names(tlm$Fouriers))))
  }
  else if(meanFun == "linear"){
    ActiveVariableIndices <- lapply(lm.list, function(tlm) which(names(tData)%in%additionalVariables))
  }
  else if(meanFun == "spec"){
    ActiveVariableIndices <- lapply(Active, function(tlm) which(names(tData)%in%tlm))
  }
  ###Prepare the mean functions for MOGP### 
  #mean_func.list.MGP <- lapply(lm.list, function(e) FormulaToString(formula(e$linModel)))
  
  #### Instead - set as constant mean for each ####
  mean_func.list.MGP <- lapply(lm.list, function(e) FormulaToStringConstant(formula(e$linModel)))
  
  ###Establish the priors for the emulators###
  Priors <- lapply(1:HowManyEmulators, function(l) GetPriorsConstant(lm.list[[l]], d=d, Choices[[l]], ActiveVariableIndices[[l]]))
  
  ###Establish the kernel types for MOGP###
  if(length(kernel) == 1) {
    Kernels <- lapply(1:HowManyEmulators, function(l) GetKernel(kernel))
  } else {
    Kernels <- lapply(1:HowManyEmulators, function(l) GetKernel(kernel[l])) 
  }
  
  #### Need to subtract linear model from targets ####
  if (HowManyEmulators == 1){
    targets <- targets - predict(lm.list[[1]]$linModel, tData)
  }
  else {
    for (cc in 1:HowManyEmulators){
      targets[cc,] <- targets[cc,] - predict(lm.list[[cc]]$linModel, tData)
    }
  }
  
  ###Establish and fit the MOGP###
  Emulators <- mogp_emulator$MultiOutputGP(inputs, targets, mean = mean_func.list.MGP,
                                           priors = Priors, inputdict = inputdict,
                                           nugget = lapply(Choices,function(e) e$Nugget), 
                                           kernel = Kernels)
  Emulators <- mogp_emulator$fit_GP_MAP(Emulators)
  
  ###Prepare return objects###
  Design <- tData[,1:(lastCand-1), drop = FALSE]
  fitting <- list(lm.object = lm.list,
                  Design = Design, 
                  ActiveIndices = ActiveVariableIndices,
                  PriorChoices = Choices)
  
  return(list(mogp = Emulators, # call mogp
              fitting.elements= fitting))
}


FormulaToStringConstant <- function(tformula){
  #'@description Parse and R formula into a string
  #'@param tformula an R formula of the form y~I(x1)+I(x_2)+ etc
  #'@return A string version to be passed into MOGP
  f = as.character(tformula)
  paste(f[2], f[1], 1, sep="")
}

GetPriorsConstant <- function(lm.emulator, d, Choices, ActiveVariables){
  #'@description This function constructs a subjective prior for the parameters of a GP emulator to be fit by MO_GP. 
  #'@param lm.emulator A lm emulator list (see AutoLMCode). The only required element of this list is the linModel component. lm.emulator$linModel is an lm object fitted to the target data. Custom lm objects can be specified using lm.emulator = list(linModel=lm(...), entering your own formulae). It is suggested that this is done elsewhere and passed here.
  #'@param d the number of input parameters
  #'@param Choices A list containing hyperprior choices that control the subjective prior.
  #'@param NonInformativeRegression If TRUE, a uniform prior is used for the regression parameters.
  #'@param NonInformativeCorrelationLengths If TRUE, a uniform prior is used for the correlation parameters. 
  #'@param NonInformativeSigma If TRUE, a uniform prior is used for the sigma parameter.
  #'@param NonInformativeNugget If TRUE, a uniform prior is used for the regression parameters.
  #'@param BetaRegressMean Prior mean for the regression coefficients. The intercept is given a uniform prior, all other regression terms get a Normal prior with mean BetaRegressMean and variance "BetaRegressSigma"
  #'@param BetaRegressSigma Prior variance for the regression coefficients.
  #'@param DeltaActiveMean The mean of a lognormal prior for the active inputs (see details). 
  #'@param DeltaActiveSigma The variance of a lognormal prior for the active inputs (see details).
  #'@param DeltaInactiveMean The mean of a lognormal prior for the inactive inputs (see details).
  #'@param DeltaInactiveSigma The variance of a lognormal prior for the inactive inputs (see details).
  #'@param NuggetProportion What proportion of the data variability is suspected to be nugget. Only used if Nugget="fit"
  #'@param Nugget either a number or "fit", "fixed" or "adaptive". This is seen by MOGP which will only fit the nugget (and hence require a prior) if "fit" is chosen. If the others, the nugget is fixed to the value given or fitted to avoid numerical instabilities adaptively. (See MOGP documentation.)
  #'@param ActiveVariables Indices indicating which parameters in the data are active
  #'@return A list of priors in the following order. A prior for the intercept (defaults to NULL), p priors for the regression coefficients, d priors for the correlation lengths, a prior for sigma squared and a prior for the nugget (NULL if Choices$Nugget != "fit")
  #'@details The linear mean has p coefficients not associated with the intercept. Each coefficient is Beta ~ Normal(BetaRegressMean, BetaRegressSigma). Our default is based on mean 0 variance 10 and is weakly informative.
  #'@details Input parameters are classified either as active or inactive. These definitions depend on whether the parameters were highlighted by preliminary fitting algorithms in our linear modelling code (see AutoLMcode) and whether the user asks for them specifically to be included (so note they may do nothing, but still get an "active" prior. There are 2 classes of prior we use for the correlation lengths. The first is for the "active" ones: log(delta) ~ Normal(DeltaActiveMean, DeltaActiveSigma) (so that delta is lognormal). default this to N(0,0.5) on the basis that we expect there to be correlation and so that they are not too long (we penalise the rigde on the likelihood surface). Inactive parameters get the same type of prior with very strong default values N(5,0.005), giving the whole distribtion at contributions to the correlation near 1. As our covariance functions are separable, the inactive variables therefore don't alter the fit. All values for these parameters are controllable by the user.
  #'@details. Sigma^2 ~ InvGamma(M,V) where we use a parameterisation of the inverse gamma based on the mode and variance. The mode is chosen so reflect the variance we can explain with simple linear fits and the variance is set using invgamMode with a bound based on the total variability in the original data. The idea behind the prior is that we allow Sigma^2 to be as large as the variance of the data so that our emulator explains none of the variability, but we expect to explain as much as could be explained by a preliminary fit of a basic model.
  #'@details The nugget distribution, if required, is found as with sigma but using the choice NuggetProportion to reflect what percentage of variability we might expect to be nugget. We may add the ability to add a user prior here. Only really important for internal variability models like climate models. Most deterministic models should use "adaptive" or "fixed" nugget. 
  p <- 0
  Priors <- lapply(1:(1+p+d+1+1), function(e) NULL)
  if(!is.null(Choices$intercept))
    print("NULL intercept fitted by default: change code")
  
  Priors[[1]] <- mogp_priors$NormalPrior(0., 0.0001)
  #Regression priors
  #if(!(Choices$NonInformativeRegression)){
  #  Betas <- lapply(1:p, function(e) mogp_priors$NormalPrior(Choices$BetaRegressMean, Choices$BetaRegressSigma))
  #  for(i in 2:(p+1)){#the first element is NULL for the intercept term
  #    Priors[[i]] <- Betas[[i-1]]
  #  }
  #}
  #Correlation length priors
  if(!(Choices$NonInformativeCorrelationLengths)){
    Deltas <- lapply(1:d, function(k) {if(k %in% ActiveVariables) 
    {mogp_priors$NormalPrior(Choices$DeltaActiveMean,Choices$DeltaActiveSigma)} 
      else {mogp_priors$NormalPrior(Choices$DeltaInactiveMean,Choices$DeltaInactiveSigma)}})
    for(j in (p+2):(p+1+d)){
      Priors[[j]] <- Deltas[[j-p-1]]
    }
  }
  #Sigma and nugget priors
  ModeSig <- var(lm.emulator$linModel$residuals)
  boundSig <- ModeSig/(1-summary(lm.emulator$linModel)$r.squared)
  if(!(Choices$NonInformativeSigma)){
    SigmaParams <- invgamMode((1-Choices$NuggetProportion)*boundSig,ModeSig)
    Sigma <- mogp_priors$InvGammaPrior(SigmaParams$alpha,SigmaParams$beta)
    Priors[[d+p+2]] <- Sigma
  }
  if(Choices$Nugget=="fit"){
    if(!Choices$NonInformativeNugget){
      NuggetParams <- invgamMode(Choices$NuggetProportion*boundSig ,Choices$NuggetProportion*ModeSig)
      Nugget <- mogp_priors$InvGammaPrior(NuggetParams$alpha, NuggetParams$beta)
      Priors[[d+p+3]] <- Nugget
    }
  }
  return(Priors)
}


PredictMOGP <- function(emulator, design){
  n_em <- emulator$mogp$n_emulators
  gp_pred <- emulator$mogp$predict(as.matrix(design), deriv=FALSE)
  for (i in 1:n_em){
    gp_pred$mean[i,] <- gp_pred$mean[i,] + predict(emulator$fitting.elements$lm.object[[i]]$linModel, design)
  }
  return(gp_pred)
}

ValidateMOGP <- function(emulator, ValidationData, which.emulator=1, IndivPars = FALSE, SDs = 2){
  require(sfsmisc)
  
  preds <- PredictMOGP(emulator, ValidationData[,1:(which(colnames(ValidationData) == 'Noise') - 1)])
  response <- ValidationData[,which(colnames(ValidationData) == 'Noise') + which.emulator]
  
  preds$upper95 <- preds$mean + SDs*sqrt(preds$unc)
  preds$lower95 <- preds$mean - SDs*sqrt(preds$unc)
  upp <- max(c(preds$upper95[which.emulator,], response))
  low <- min(c(preds$lower95[which.emulator,], response))
  
  if (IndivPars == TRUE){
    p <- length(emulator$fitting.elements$ActiveIndices[[which.emulator]]) + 1
    if(p<2){
      par(mfrow = c(1, 1), mar=c(4, 4, 1, 1))
    }
    else if(p<3){
      par(mfrow = c(1, 2), mar=c(4, 4, 1, 1))
    }
    else if(p<4){
      par(mfrow = c(1, 3), mar=c(4, 4, 1, 1))
    }
    else if(p <5){
      par(mfrow = c(2, 2), mar=c(4, 4, 1, 1))
    }
    else if(p<7){
      par(mfrow = c(2, 3), mar=c(4, 4, 1, 1))
    }
  }
  errbar(preds$mean[which.emulator,], preds$mean[which.emulator,], preds$upper95[which.emulator,], preds$lower95[which.emulator,], cap = 0.015, pch=20, 
         ylim=c(low,upp), main="",xlab = "Prediction",ylab="Data")
  points(preds$mean[which.emulator,], response, pch=19,
         col = ifelse(response > preds$upper95[which.emulator,] | response < preds$lower95[which.emulator,], "red", "green"))
  if (IndivPars == TRUE){
    active <- emulator$fitting.elements$ActiveIndices[[which.emulator]]

    for (i in 1:length(active)){
      errbar(ValidationData[,active[i]], preds$mean[which.emulator,], preds$upper95[which.emulator,], preds$lower95[which.emulator,], cap = 0.015, pch=20, 
             ylim=c(low,upp), xlab = "Input",ylab="Prediction", main = paste(colnames(ValidationData)[active[i]]))
      points(ValidationData[,active[i]], response, pch=19,
             col = ifelse(response > preds$upper95[which.emulator,] | response < preds$lower95[which.emulator,], "red", "green"))
    }
  }
}

