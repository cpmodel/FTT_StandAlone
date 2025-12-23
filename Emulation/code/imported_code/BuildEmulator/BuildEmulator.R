twd <- getwd()
packages <- c('GenSA', 'far', 'fields', 'lhs', 'maps', 'mco', 'mvtnorm', 
              'ncdf4', 'parallel', 'rstan', 'shape', 'tensor', 'withr', 
              'loo', 'bayesplot','MASS')
sapply(packages, require, character.only = TRUE, quietly = TRUE)

source("BuildEmulator/AutoLMcode.R")
source("BuildEmulator/CustomPredict.R")
source("BuildEmulator/JamesNewDevelopment.R")
source("BuildEmulator/DannyDevelopment.R")

# # Stan files for GP Emulator in Stan
# tfile_loc = paste(twd, "/BuildEmulator/FitGP.stan", sep = "")
# tprednewfile_loc = paste(twd, "/BuildEmulator/PredictGen.stan", sep = "")
# 
# ccode_fit <- stanc(file = tfile_loc)
# model_fit <-stan_model(stanc_ret = ccode_fit)
# 
# ccode_predict <- stanc(file = tprednewfile_loc)
# model_predict <- stan_model(stanc_ret = ccode_predict)
# 
# gpstan.default.params <- list(SigSq = 0.15, SigSqV = 0.25, AlphaAct = 4, BetaAct = 4,
#                               AlphaInact = 42, BetaInact = 9, AlphaNugget = 3,
#                               BetaNugget = 0.1, AlphaRegress = 0, BetaRegress = 10, 
#                               nugget = 0.0001, SwitchDelta = 1, SwitchNugget = 1, SwitchSigma = 1)

EMULATE.lm <- function(Response, tData, dString="tData",maxdf=NULL,tcands=cands,tcanfacs=canfacs,TryFouriers=FALSE,maxOrder=NULL){
#' Generate a linear model for Gaussian Process (GP) Emulator
#' 
#' @param Response a string corresponding to the response variable
#' @param tData a data frame containing the inputs and outputs
#' @param dString a string corresponding to the name of a data frame
#' @param maxdf a maximim number of degrees of freedom that we are expecting to 
#' lose by adding terms to the linear model
#' @param tcands a vector of parameter names
#' @param tcanfacs a vector of parameter names that are factors
#' @param TryFouriers a logical argument with default FALSE. TRUE allows
#' the Fourier transformation of the parameters
#' @param maxOrder maximum order of Fourier terms (Fourier series)
#' 
#' @return A linear model to EMULATE.gpstan function
#' 
    if(is.null(maxdf)){
    maxdf <- ceiling(length(tData[,1])/10)
  }
  startLM <- eval(parse(text=paste("lm(",Response,"~1,data=",dString,")",sep="")))
  if(maxdf < 2){
    return(list(linModel=startLM,Names=NULL,mainEffects=NULL,Interactions=NULL,Factors=NULL,FactorInteractions=NULL,ThreeWayInters=NULL,DataString=dString,ResponseString=Response,tData=tData,BestFourier=TRUE))
  }
  if(TryFouriers){
    msl <- list(linModel=startLM,Names=NULL,mainEffects=NULL,Interactions=NULL,Factors=NULL,FactorInteractions=NULL,ThreeWayInters=NULL,DataString=dString,ResponseString=Response,tData=tData,BestFourier=TRUE,maxOrder=maxOrder)
  }
  else{
    msl <- list(linModel=startLM,Names=NULL,mainEffects=NULL,Interactions=NULL,Factors=NULL,FactorInteractions=NULL,ThreeWayInters=NULL,DataString=dString,ResponseString=Response,tData=tData,BestFourier=FALSE)
  }
  added <- AddBest(tcands,tcanfacs,msl)
  for(i in 1:30){
    added <- AddBest(tcands,tcanfacs,added)
    if(!is.null(added$Break))
      break
  }
  print(summary(added$linModel))
  if(length(tData[,1])-1-added$linModel$df > maxdf){
    trm <- removeNterms(N=500,linModel=added$linModel,dataString=added$DataString,responseString=added$ResponseString,Tolerance=sum(sort(anova(added$linModel)$"Sum Sq"))*(1e-4),Names=added$Names,mainEffects=added$mainEffects,Interactions=added$Interactions,Factors=added$Factors,FactorInteractions=added$FactorInteractions,ThreeWayInters=added$ThreeWayInters,tData=added$tData,Fouriers = added$Fouriers)
    print(summary(trm$linModel))
    trm2 <- removeNterms(N=max(c(0,length(tData[,1])-maxdf-1-trm$linModel$df)),linModel=trm$linModel,dataString=added$DataString,responseString=added$ResponseString,Tolerance=sum(sort(anova(added$linModel)$"Sum Sq"))*(5e-1),Names=trm$Names,mainEffects=trm$mainEffects,Interactions=trm$Interactions,Factors=trm$Factors,FactorInteractions=trm$FactorInteractions,ThreeWayInters=trm$ThreeWayInters,tData=added$tData,Fouriers=trm$Fouriers)
    print(summary(trm2$linModel))
    trm2$pre.Lists <- get.predict.mats(trm2$linModel)
    trm2$DataString <- dString
    trm2$ResponseString <- Response
    return(trm2)
  }
  else{
    return(added)
  }
}

CovMatrix <- function(Design, cls){
#' Constructs a squared exponential correlation matrix
#' 
#' @param Design a matrix of inputs for design set
#' @param cls a vector of correlation length parameters
#' 
#' @return A matrix of correlation calculated for design points
#' 
  n = dim(Design)[1] # number of training points
  if(is.null(n))
    n <- length(Design)
  p = dim(Design)[2] # number of parameters
  if(is.null(p))
    p <- 1
  stopifnot(p == length(cls))
  if(!is.numeric(as.matrix(Design))){
    for (i in 1:p){
      Design[,i] = as.numeric(as.character(Design[,i]))
    }
  }
  D = as.matrix(dist(scale(Design,center=FALSE,scale=cls),method="euclidean",diag=TRUE,upper=TRUE))
  return(exp(-(D^2)))
}

NewCov <- function(x, Design, cls){
#' A matrix of correlations (squared exponential form) between x and Design 
#' matrices
#' 
#' @param x a matrix of inputs for validation set
#' @param Design a matrix of inputs for design set
#' @param cls a vector of correlation length parameters
#' 
#' @return A matrix of correlations between x and Design
#' 
  n = dim(Design)[1] # number of training points
  if(is.null(n))
    n <- length(Design)
  m = dim(x)[1] 
  p = dim(Design)[2] # number of parameters
  if(is.null(p))
    p <- 1
  stopifnot(p == length(cls))
  if(!(is.null(names(Design))))
    x = x[,names(Design)]
  if(!is.numeric(as.matrix(Design))){
    for (i in 1:p){
      Design[,i] = as.numeric(as.character(Design[,i]))
    }
  }
  if(!is.numeric(as.matrix(x))){
    for (i in 1:p){
      x[,i] = as.numeric(as.character(x[,i]))
    }
  }
  D = as.matrix(rdist(scale(Design,center=FALSE,scale=cls),scale(x,center=FALSE,scale=cls))) 
  return(exp(-(D^2)))
}

#Build a GP stan emulator
#meanResponse is the output of EMULATE.lm containing a linear model emulator
#... Refer to covariance parameters

EMULATE.gpstan <- function(meanResponse, CompiledModelFit = model_fit, sigmaPrior = FALSE, nuggetPrior = FALSE,
                           activePrior = FALSE, activeVariables = NULL, tData, additionalVariables = NULL, FastVersion = FALSE, 
                           prior.params = gpstan.default.params, ...){
#' Function implement Gaussian process (GP) emulator
#' 
#'  @param meanResponse the output of EMULATE.lm containing a linaer model emulator
#'  @param CompiledModelFit an instance of S4 class stanmodel used for fitting a GP model
#'  @param sigmaPrior a logical argument with FALSE (default) specifying sigma prior parameters
#'  at the values found from the linear model fit (NEED MORE CLARIFICATION)
#'  @param nuggetPrior a logical argument with FALSE (default) specifying nugget parameter at
#'  fixed value. TRUE specifying a prior distribution for the nugget parameter
#'  @param activePrior a logical argument with FALSE (default) specifying the same prior 
#'  for correlation length parameters. TRUE specifying different priors for correlation 
#'  length parameters.
#'  @param activeVariables a vector of parametere names considered to be active, with default NULL
#'  @param tData a data frame containing the inputs and outputs and must be the same as that to create meanResponse 
#'  @param additionalVariables is a vector of parameter names that we want to fit a GP to that didn't make it into the model
#'  the default model, with additionalVariables=NULL, is to only treat those terms that make it into the linear model as active
#'  @param FastVersion TRUE value results at saving parameter values at the posterior mean
#'  FALSE (default) saves the posterior samples for parameters. 
#'  @param prior.params a list of parameters to the prior specification. The default is 
#'  gpstan.default.params. 

#'  @return A GP emulator object

    #Design <- as.matrix(tData[,which((names(tData)%in%meanResponse$Names) | (names(tData)%in%meanResponse$Factors) | (names(tData)%in%additionalVariables) | (names(tData) %in% names(meanResponse$Fouriers)))])
  Design <- as.matrix(tData[,which((names(tData)%in%meanResponse$Names) |  (names(tData)%in%additionalVariables) | (names(tData) %in% names(meanResponse$Fouriers)))])
  colnames(Design) <- names(tData)[which((names(tData)%in%meanResponse$Names) |  (names(tData)%in%additionalVariables) | (names(tData) %in% names(meanResponse$Fouriers)))]
  #Perhaps can take factors out of Design? H still includes them, only effects corr lengths in Stan
  #Slightly careful handling in EMULATOR.gpstan would be required
  tF <- tData[,which(names(tData)==meanResponse$ResponseString)]
  H <- model.matrix(meanResponse$linModel)
  N1 <- dim(Design)[1]
  Np <- dim(H)[2]
  if(sigmaPrior) prior.params$SwitchSigma = 2
  if(nuggetPrior) {
    prior.params$SwitchNugget = 2
    #prior.params$UpperLimitNugget = prior.params$BetaNugget/(prior.params$AlphaNugget-1) + 10*sqrt(prior.params$BetaNugget^2/((prior.params$AlphaNugget-1)^2*(prior.params$AlphaNugget-2)))
    } 
#  else {
#    prior.params$UpperLimitNugget = 2*prior.params$nugget
#     }
  if(activePrior) prior.params$SwitchDelta = 2
  if(!sigmaPrior) {
    consEm <- EMULATE.lm(Response=meanResponse$ResponseString, tData=tData, tcands="Noise",tcanfacs=NULL,TryFouriers=TRUE,maxOrder=2,maxdf = 0)
    sd2 <- summary(consEm$linModel)$sig - summary(meanResponse$linModel)$sig
    sigsq <- summary(meanResponse$linModel)$sigma
    sigsqvar <- sd2
    # consider the constant Gaussian Process mean
    if(Np == 1) sigsqvar <- summary(meanResponse$linModel)$sig
    prior.params$SigSq <- sigsq
    prior.params$SigSqV <- sigsqvar
  }
  if(activePrior) {
    #if(is.null(activeVariables)) active.inputs <- names(tData)[which((names(tData)%in%meanResponse$Names) | (names(tData)%in%additionalVariables) |(names(tData)%in%names(meanResponse$Fouries)))]
    #else active.inputs <- activeVariables
    #inactive.inputs <- names(tData)[-which((names(tData)%in%active.inputs)|(names(tData)%in%meanResponse$ResponseString))]
    active.inputs <- colnames(Design)[colnames(Design)%in%activeVariables]
    inactive.inputs <- colnames(Design)[-which(colnames(Design)%in%active.inputs)]
    
    Design.active <- as.matrix(tData[, active.inputs])
    Design.inactive <- as.matrix(tData[, inactive.inputs])
    Design <- cbind(Design.active, Design.inactive)
    colnames(Design) <- c(active.inputs, inactive.inputs)
    p.active <- length(active.inputs)
    p.inactive <- length(inactive.inputs)
    p <- dim(Design)[2]
    init.list <- list(list(beta=array(meanResponse$linModel$coefficients, dim = Np), sigma=prior.params$SigSq, nugget = prior.params$nugget, 
                           delta_par=array(c(rep(0.05, p.active), rep(0.7, p.inactive)), dim=p)),
                      list(beta=array(meanResponse$linModel$coefficients, dim = Np), sigma=prior.params$SigSq, nugget = prior.params$nugget, 
                           delta_par=array(c(rep(0.1, p.active), rep(1, p.inactive)), dim=p)))
    StanEmulator <- sampling(CompiledModelFit, data = list(N1 = N1, pact = p.active, pinact = p.inactive, 
                                                           p = p, Np = Np, SwitchDelta = prior.params$SwitchDelta, 
                                                           SwitchNugget = prior.params$SwitchNugget, SwitchSigma = prior.params$SwitchSigma, 
                                                           SigSq = prior.params$SigSq, SigSqV = prior.params$SigSqV, 
                                                           AlphaAct = prior.params$AlphaAct, BetaAct = prior.params$BetaAct, 
                                                           AlphaInact = prior.params$AlphaInact, BetaInact = prior.params$BetaInact, 
                                                           AlphaNugget = prior.params$AlphaNugget, BetaNugget = prior.params$BetaNugget, 
                                                           AlphaRegress = prior.params$AlphaRegress, BetaRegress = prior.params$BetaRegress, 
                                                           nuggetfix = prior.params$nugget, 
                                                           X1 = Design, y1 = tF, H1 = H),
                                                           #UpperLimitNugget = prior.params$UpperLimitNugget), 
                             iter = 2000, warmup = 1000, chains = 2, cores = 2, init = init.list,
                             pars = c('nugget', 'sigma', 'delta_par', 'beta', 'log_lik'), ...)
  } else {
    # consider the same prior specification for delta_par (correlation length parameter)
    Design <- as.matrix(tData[,which((names(tData)%in%meanResponse$Names) |  (names(tData)%in%additionalVariables) | (names(tData) %in% names(meanResponse$Fouriers)))])
    colnames(Design) <- names(tData)[which((names(tData)%in%meanResponse$Names) |  (names(tData)%in%additionalVariables) | (names(tData) %in% names(meanResponse$Fouriers)))]
    p <- dim(Design)[2]
    p.active = p.inactive = 1
    init.list <- list(list(beta=array(meanResponse$linModel$coefficients, dim = Np), sigma=prior.params$SigSq, nugget = prior.params$nugget, delta_par=array(rep(0.7, p), dim=p)),
                      list(beta=array(meanResponse$linModel$coefficients, dim = Np), sigma=prior.params$SigSq, nugget = prior.params$nugget, delta_par=array(rep(1, p), dim=p)))
    StanEmulator <- sampling(CompiledModelFit, data = list(N1 = N1, pact = p.active, pinact = p.inactive, 
                                                           p = p, Np = Np, SwitchDelta = prior.params$SwitchDelta, 
                                                           SwitchNugget = prior.params$SwitchNugget, SwitchSigma = prior.params$SwitchSigma, 
                                                           SigSq = prior.params$SigSq, SigSqV = prior.params$SigSqV, 
                                                           AlphaAct = prior.params$AlphaAct, BetaAct = prior.params$BetaAct, 
                                                           AlphaInact = prior.params$AlphaInact, BetaInact = prior.params$BetaInact, 
                                                           AlphaNugget = prior.params$AlphaNugget, BetaNugget = prior.params$BetaNugget,  
                                                           AlphaRegress = prior.params$AlphaRegress, BetaRegress = prior.params$BetaRegress, 
                                                           nuggetfix = prior.params$nugget, #UpperLimitNugget = prior.params$UpperLimitNugget,
                                                           X1 = Design, y1 = tF, H1 = H), 
                             iter = 2000, warmup = 1000, chains = 2, cores = 2, init = init.list,
                             pars = c('nugget', 'sigma', 'delta_par', 'beta', 'log_lik'), ...)
  }
  ParameterSamples <- rstan::extract(StanEmulator, pars = c('sigma', 'delta_par', 'beta', 'nugget'))
  if(FastVersion){
    lps <- extract_log_lik(StanEmulator)
    tMAP <- which.max(rowSums(lps))
    A <- ParameterSamples$sigma[tMAP]^2*CovMatrix(Design, ParameterSamples$delta_par[tMAP, ]) #+ diag(ParameterSamples$nugget[tMAP], nrow = dim(Design)[1], ncol = dim(Design)[1])
    QA <- chol(A)
    diff <- tF - H%*%ParameterSamples$beta[tMAP, ]
    Ldiff <- backsolve(QA, diff, transpose=TRUE) #part of the mean update that can be done offline
    FastParts <- list(tMAP=tMAP, A=A, QA=QA, Ldiff=Ldiff)
  }
  else{
    FastParts <- NULL
  }
  gp.list <- list(Design=Design, tF=tF, H=H, ParameterSamples=ParameterSamples, FastParts=FastParts, StanModel = StanEmulator, 
                  prior.params = prior.params, init.list = init.list)
  return(c(meanResponse,gp.list))
}



EMULATOR.gpstan <- function(x, Emulator, GP=TRUE, FastVersion=FALSE,  CompiledModelPredict = model_predict){
#' Predict method for Gaussian Process model
#' 
#' @param x a matrix of inputs
#' @param Emulator a GP emulator from EMULATE.gpstan function
#' @param GP not sure about this argument (?)
#' @param FastVersion TRUE value results at saving parameter values 
#' at the posterior mean; FALSE (default) saves the posterior samples for parameters
#' @param CompiledModelPredict an instance of S4 class stanmodel that is used to produce predictions
#' 
#' @return A list contatining 'Expectation' and 'Variance'
#' 
  if(!GP){# || (Emulator$nugget==1)){
    lin <- Emulator$linModel
    newDat <- as.data.frame(x)
    prediction <- try(custom.predict(lin,newdata=newDat,Emulator$pre.Lists),silent=TRUE)
    if(inherits(prediction,"try-error")){
      stop("Ensure the data frame has the correct named columns")
    }
    return(list(Expectation=as.vector(prediction[,1]), Variance = as.vector(prediction[,2])^2))
  }
  else{
    Xpred <- as.data.frame(x)
    tt = terms(Emulator$linModel)
    Terms = delete.response(tt)
    mm = model.frame(Terms, Xpred, xlev = Emulator$linModel$xlevels)
    Hpred = model.matrix(Terms, mm, contrasts.arg = Emulator$linModel$contrasts)
    #Ensure Xpred enters the emulator with the columns in the exact same order as when
    #emulated to ensure correlation lengths mean the same thing.
    if(dim(Emulator$Design)[2] > 1) {
      tinds = sapply(1:length(colnames(Emulator$Design)), function(k) which(colnames(Emulator$Design)[k] == colnames(Xpred)) )
      Xpred <- Xpred[, tinds]
    }else {
      Xpred <- matrix(Xpred[, 1], ncol = 1)
    }  
    if(FastVersion){
      if(is.null(Emulator$FastParts))
        stop("Build a FastVersion with EMULATE.gpstan first")
      Covtx <- Emulator$ParameterSamples$sigma[Emulator$FastParts$tMAP]^2*NewCov(Xpred, Design=Emulator$Design, cls = Emulator$ParameterSamples$delta[Emulator$FastParts$tMAP,])
      txtA <- backsolve(Emulator$FastParts$QA,Covtx,transpose=TRUE)
      if(dim(Emulator$H)[2]> 1) tExpectation <- as.vector(Hpred%*%Emulator$ParameterSamples$beta[Emulator$FastParts$tMAP,] + crossprod(txtA, Emulator$FastParts$Ldiff))
      else tExpectation <- as.vector(Hpred%*%Emulator$ParameterSamples$beta[Emulator$FastParts$tMAP] + crossprod(txtA, Emulator$FastParts$Ldiff))
      tVariance <- as.vector(diag(diag(Emulator$ParameterSamples$sigma[Emulator$FastParts$tMAP]^2,nrow=dim(Xpred)[1],ncol=dim(Xpred)[1]) - crossprod(txtA)))
      tVariance[which(tVariance < 0)] <- 0
      StandardDev <- sqrt(tVariance)
    }
    else{
      fit.y2 <- sampling(CompiledModelPredict, 
                         data = list(X1 = Emulator$Design, X2 = Xpred, y1= Emulator$tF, 
                                     H1 = Emulator$H, H2 = Hpred, N1 = dim(Emulator$Design)[1], N2 = dim(Xpred)[1],  
                                     Np = dim(Emulator$H)[2], p = dim(Emulator$Design)[2], M = dim(Emulator$ParameterSamples$beta)[1], 
                                     sigma = Emulator$ParameterSamples$sigma, beta = Emulator$ParameterSamples$beta, 
                                     delta = Emulator$ParameterSamples$delta_par, nugget = Emulator$ParameterSamples$nugget),
                         iter = 1, warmup = 0, chains = 1, cores = 1, pars = c("tmeans","tsds"), include=TRUE, 
                         algorithm = c('Fixed_param'))
      predict.y2 <- extract(fit.y2, pars = c('tmeans','tsds'))
      tExpectation <- predict.y2$tmeans[1, ]
      StandardDev <- predict.y2$tsds[1, ]
    }
    return(list(Expectation = tExpectation, Variance=StandardDev^2))
  }
}

EMULATOR.gpstan.multicore <- function(x, Emulator, GP=TRUE, FastVersion=FALSE, batches=500){
#' Predict method for Gaussian process model for many runs
#' 
#' @param x a matrix of  
#' @param Emulator a GP emulator from EMULATE.gpstan function
#' @param GP not sure about this argument (?)
#' @param FastVersion TRUE value results at using parameter values 
#' at the posterior mean; FALSE (default) uses the posterior samples for parameters

#' 
#' @return A list contatining 'Expectation' and 'Variance'
#' 
  
  Nruns <- dim(x)[1]
  Nbatch <- ceiling(Nruns/batches)
  batchSeq <- lapply(1:(Nbatch-1), function(k) seq(from=1+(k - 1)*batches,by=1,length.out = batches))
  batchSeq[[Nbatch]] <- batchSeq[[Nbatch-1]][batches]:Nruns
  tSamples <- mclapply(batchSeq, function(e) EMULATOR.gpstan(x=x[e,], Emulator=Emulator, GP=GP, FastVersion = FastVersion),mc.cores = 6)
  tExpectation <- rep(NA,Nruns)
  tVariance <- rep(NA,Nruns)
  for(i in 1:Nbatch){
    tExpectation[batchSeq[[i]]] <- tSamples[[i]]$Expectation
    tVariance[batchSeq[[i]]] <- tSamples[[i]]$Variance
  }
  return(list(Expectation=tExpectation,Variance=tVariance))
}

ExtractCentredScaledDataAndBasis <- function(OriginalField, scaling=1){
  #MeanField is now the uncentered, unscaled field we want to emulate. The last step is to centre, scale and extract the basis
  EnsMean <- apply(OriginalField, MARGIN=1, mean)
  CentredField <- OriginalField
  for(i in 1:dim(OriginalField)[2]){
    CentredField[,i] <- CentredField[,i] - EnsMean
  }
  CentredField <- CentredField/scaling
  Basis <- svd(t(CentredField))$v
  return(list(tBasis=Basis, CentredField = CentredField, EnsembleMean = EnsMean, scaling=scaling))
}


BuildStanEmulator <- function(Response, tData, cands, additionalVariables=NULL, 
                              canfacs = NULL, TryFouriers = TRUE, maxOrder = 2, maxdf = NULL, 
                              CompiledModelFit = model_fit, 
                              sigmaPrior = FALSE, nuggetPrior=FALSE, activePrior = FALSE, 
                              activeVariables = NULL, prior.params = gpstan.default.params, 
                              FastVersion = TRUE){
#' Function to construct Gaussian Proceess emulator in one go
#' 
#' @param Response a character corresponding to the response of interest
#' @param tData a data frame of inputs and a vector of output responses
#' @param cands a vector of input parameters
#' @param additionalVariables a vector of parameter names that we want to fit a GP to
#' that didn't make it into the model
#' @param canfacs a vector of input parameters that are factors
#' @param TryFouriers a logical argument with TRUE (default) allowing the Fourier
#' transformation of input parameters
#' @param maxOrder maximum order of Fourier terms (Fourier series)
#' @param maxdf maximum degrees of freedom
#' @param CompiledModelFit an instance of S4 class stanmodel used for fitting a GP model
#' @param sigmaPrior a logical argument with FALSE (default) specifying sigma prior parameters
#' at the values found from the linear model fit
#' @param nuggetPrior a logical argument with FALSE (default) specifying nugget parameter
#' at fixed value and TRUE specifying a prior distribution for the nugget parameter.
#' @param activePrior a logical argument with FALSE (default) specifying the same prior 
#' for correlation length parameters. TRUE specifying different priors for correlation length
#' parameters.
#' @param activeVariables a vector of parameter names considered to be active, with default NULL
#' @param prior.params default parameters to the prior specification. The default is
#' gpstan.default.params.
#' @param FastVersion TRUE value results at saving parameter values at posterior mean, 
#' FALSE (default) saves the posterior samples for parameters.
#' 
#' @return A GP Emulator object
#' 
  myem.lm <- EMULATE.lm(Response=Response, tData=tData, tcands=cands,
                        tcanfacs=canfacs,TryFouriers=TryFouriers,maxOrder=maxOrder,
                        maxdf = maxdf)
  myem.gp = EMULATE.gpstan(meanResponse=myem.lm, sigmaPrior = sigmaPrior, 
                           nuggetPrior = nuggetPrior, activePrior = activePrior, 
                           activeVariables = activeVariables, tData = tData,
                           additionalVariables=additionalVariables, FastVersion=FastVersion,
                           prior.params = prior.params)
  myem.gp
}

#This function must have tData with order: Parameters, Noise, Coefficients
InitialBasisEmulators <- function(tData, HowManyEmulators, additionalVariables=NULL, sigmaPrior = FALSE, 
                                  nuggetPrior = FALSE, activePrior = FALSE, activeVariables = NULL, 
                                  prior.params = gpstan.default.params, ...){
  lastCand <- which(names(tData)=="Noise")
  tfirst <- lastCand + 1
  if(is.null(HowManyEmulators))
    HowManyEmulators <- length(names(tData)) - lastCand
  lapply(1:HowManyEmulators, function(k) try(BuildStanEmulator(Response=names(tData)[lastCand+k], tData=tData, cands=names(tData)[1:lastCand], 
                                                               additionalVariables=additionalVariables[[k]], maxdf=ceiling(length(tData[,1])/10)+1, 
                                                               sigmaPrior = sigmaPrior, nuggetPrior = nuggetPrior, activePrior = activePrior, 
                                                               activeVariables = activeVariables, prior.params = prior.params, ...), silent = TRUE))
}

ValidPlotNew <- function(fit, x, y, ObsRange=FALSE, ...){
  #' @param fit a data frame of emulator predictions. First column corresponds
  #' to the posterior mean, second and third columns correspond to the lower
  #' and upper quantiles (to be plotted) respectively.
  #' @param x values of the inputs to be plotted on the x axis
  #' @param y a vector of simulator evaluations at x
  #' @param ObsRange. Boolean to indicate if the last row of fit represents the
  #' observations and should not be plotted, but used to expand the ranges
  #' @param ... usual arguments to plotting functions. 
  inside <- c()
  outside <- c()
  for(i in 1:length(y)) {
    if(fit[i, 2] <= y[i] & y[i] <= fit[i, 3]) {
      inside <- c(inside,i)
    }
  }
  outside <- c(1:length(y))[-inside]
  tRange <- range(fit)
  if(ObsRange){
    fit <- fit[-nrow(fit),]
  }
  plot(x, fit[, 1], pch=20, ylim=tRange, ...)
  arrows(x, fit[, 2], x, fit[, 3],
         length=0.05, angle=90, code=3, col='black')
  points(x[inside], y[inside], pch=20, col='green')
  points(x[outside], y[outside], pch=20, col='red')
}

ValidPlot <- function(fit, X, y, interval = c(), axis = 1, heading = " ", 
                      xrange = c(-1, 1), ParamNames) {
#' Plots the emulator predictions together with error bars and true values.
#'
#' @param fit a data frame of emulator predictions. First column corresponds
#' to the posterior mean, second and third columns correspond to the lower
#' and upper quantiles respectively.
#' @param X a data frame of inputs
#' @param y a vector of simulator evaluations at X
#' @param interval a vector to define the y axis limit of the plot, the 
#' default option is the empty vector.
#' @param axis the input against which the emulator predictions are plotted
#' @param heading a title of the plot
#' @param xrange a vector to define the x limit of the plot
#' @param ParamNames a vector of names of the parameters in the data frame
#' @param OriginalRanges. If TRUE, LOOs will be plotted on the original parameter ranges
#' Those ranges will be read from a file containing the parameter ranges and whether the 
#' parameters are logged or not. Defaults to FALSE, where parameters will be plotted on [-1,1]
#' @param RangeFile A .R file that will be sourced in order to determine the ranges of the
#' parameters to be plotted and whether they are on a log scale or not. If NULL when OrignialRanges
#' is called, a warning is thrown and the plot is given on [-1,1]
#' 
#' @return A plot of emulator predictions together with the error bars
#' (plus and minus 2*standard deviations) and true values.
  outside <- c()
  inside <- c()
  maxval <- max(fit)
  minval <- min(fit)
  for(i in 1:length(y)) {
    if(fit[i, 2] <= y[i] & y[i] <= fit[i, 3]) {
      inside <- c(inside,i)
    }
  }
  outside <- c(1:length(y))[-inside]
  chosen_column <- which(colnames(X) == ParamNames[axis])
  if(is.null(interval)) {
    plot(X[ , chosen_column], fit[, 1], pch=20, ylab='Y', 
         xlab=ParamNames[axis],
         ylim=c(minval, maxval),
         xlim=xrange, main=heading,cex.main=0.8)
  } else {
    plot(X[ , chosen_column], fit[, 1], pch=20, ylab='Y', 
         xlab=ParamNames[axis], ylim=interval,
         xlim=xrange, main=heading,cex.main=0.8)
  }
  arrows(X[, chosen_column], fit[, 2], X[, chosen_column], fit[, 3],
         length=0.05, angle=90, code=3, col='black')
  points(X[inside, chosen_column], y[inside], pch=20, col='green')
  points(X[outside, chosen_column], y[outside], pch=20, col='red')
}
virtual.LOO <- function(Design, y, cls, sigma, H, beta, nugget) {
  #' Function to compute the virtual Leave-One-Out formulas.
  #' 
  #' @param Design a matrix of inputs for design set.
  #' @param y a vector of simulator responses at Design. 
  #' @param cls a vector of correlation length parameters. 
  #' @param sigma a real value of scale parameter.
  #' @param H a matrix of regression functions evaluted at the design points.
  #' @param beta a vector of regression coefficients.
  #' @param nugget a real value of a nugget parameter. 
  #' 
  #' @return  a random generation for the normal distribution with mean and standard deviation
  #' found using the virtual Leave-One-Out formulas.
  expectation <- c()
  variance <- c()
  predict_y <- c()
  Gamma2 <- sigma^2*CovMatrix(Design, cls) + diag(nugget, nrow = dim(Design)[1])
  Gamma2Inv <- solve(Gamma2)
  MeanRes <- c(H%*%beta)
  Gamma2InvY <- Gamma2Inv %*%(y - MeanRes)
  DiagGamma2Inv <- diag(Gamma2Inv)
  for(i in 1:length(y)) {
    expectation[i] <- y[i] - Gamma2InvY[i]/DiagGamma2Inv[i]
    variance[i] <- 1/DiagGamma2Inv[i]
    predict_y[i] <- rnorm(1, mean = expectation[i], sd = sqrt(variance[i]))
  } 
  return(predict_y)
}

LOO.plot <- function(StanEmulator, ParamNames, 
                     OriginalRanges = FALSE, RangeFile=NULL, Obs=NULL, ObsErr=NULL, ObsRange=FALSE) {
  #' Function to generate Leave-One-Out predictions.
  #' 
  #' @param StanEmulator a GP emulator from EMULATE.gpstan function.
  #' @param ParamNames a vector of names for parameters.
  #' @param OriginalRanges. If TRUE, LOOs will be plotted on the original parameter ranges
  #' Those ranges will be read from a file containing the parameter ranges and whether the 
  #' parameters are logged or not. Defaults to FALSE, where parameters will be plotted on [-1,1]
  #' @param RangeFile A .R file that will be sourced in order to determine the ranges of the
  #' parameters to be plotted and whether they are on a log scale or not. If NULL when OrignialRanges
  #' is called, a warning is thrown and the plot is given on [-1,1]
  #' @param Obs. The scalar value of the observations to be plotted as a dasked line if not NULL
  #' @param ObsErr. Observation error (scalar). If this is NULL when obs is not NULL, a warning is thrown.
  #' @param OriginalRanges. If TRUE, LOOs will be plotted on the original parameter ranges
  #' Those ranges will be read from a file containing the parameter ranges and whether the 
  #' parameters are logged or not. Defaults to FALSE, where parameters will be plotted on [-1,1]
  #' @param RangeFile A .R file that will be sourced in order to determine the ranges of the
  #' parameters to be plotted and whether they are on a log scale or not. If NULL when OrignialRanges
  #' is called, a warning is thrown and the plot is given on [-1,1]
  #' @param ObsRange. Boolean to indicate if the plot window should have y ranges that include
  #' obs uncertainty (defaults to FALSE to facilitate emulator diagnostics)
  #' 
  #' @return a data frame with three columns, with first column corresponding to posterior mean, 
  #' and second and third columns corresponding to the minus and plus two standard deviations.
  #' The function also generates LOO validation plots.
  
  predict.y <- sapply(1:dim(StanEmulator$ParameterSamples$delta_par)[1], function(k)
    virtual.LOO(Design = StanEmulator$Design, y = StanEmulator$tF,
                H = StanEmulator$H, beta = StanEmulator$ParameterSamples$beta[k, ],
                cls = StanEmulator$ParameterSamples$delta_par[k, ],
                sigma = StanEmulator$ParameterSamples$sigma[k], 
                nugget = StanEmulator$ParameterSamples$nugget[k]))
  mean.predict.y <- rowMeans(predict.y)
  sd.predict.y <- apply(predict.y, 1, sd)
  fit.stan <- data.frame(cbind(mean.predict.y, mean.predict.y-2*sd.predict.y, 
                               mean.predict.y + 2*sd.predict.y))
  names(fit.stan) <- c('posterior mean', 'lower quantile', 'upper quantile')
  if(ObsRange){
    fit.stan <- rbind(fit.stan, c(Obs, Obs-2*ObsErr,Obs+2*ObsErr))
  }
  p <- length(ParamNames)
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
  else if(p<10){
    par(mfrow = c(3, 3), mar=c(4, 4, 1, 1))
  }
  else if(p<13){
    par(mfrow = c(4, 3), mar=c(4, 4, 1, 1))
  }
  else if(p<=16){
    par(mfrow = c(4, 4), mar=c(4, 4, 1, 1))
  }
  #par(mfrow = c(1, 3), mar=c(4, 4, 1, 1))
  theDesign <- StanEmulator$Design
  if(OriginalRanges){
    if(is.null(RangeFile))
      stop("Cannot plot on original ranges as no RangeFile Specified")
    else{
      tRanFile <- try(source(RangeFile), silent=TRUE)
      if(inherits(tRanFile, "try-error"))
        stop("Invalid RangeFile given")
      if(!is.null(param.names)
         & !is.null(param.lows)
         & !is.null(param.highs)
         & !is.null(param.defaults)
         & !is.null(which.logs)){
        PlotOrder <- sapply(ParamNames, function(aName) which(param.names==aName))
        #TRY LAPPLY IF THE ABOVE FAILS NEEDING NUMERIC VECTORS NOT STRINGS
        #First cut design to just ParamNames in the order of ParamNames
        DesignOrder <- sapply(ParamNames, function(aName) which(colnames(theDesign)==aName))
        #Is design order a permutation? Think so (with cut columns)
        PermutedDesign <- theDesign[,DesignOrder]
        AllLogs <- rep(FALSE,length(param.names))
        AllLogs[which.logs] <- TRUE
        param.names <- param.names[PlotOrder]
        param.lows <- param.lows[PlotOrder]
        param.highs <- param.highs[PlotOrder]
        NewLogs <- AllLogs[PlotOrder]
        which.logs <- which(NewLogs)
        theDesign <- DesignConvert(PermutedDesign, param.names = param.names, 
                                   param.lows = param.lows, param.highs = param.highs, 
                                   which.logs = which.logs)
      }
      else
        stop("Ranges file doesnt define the right variables")
    }
  }
  else{
    which.logs <- c()
  }
  tlogs <- rep("",p)
  tlogs[which.logs] <- "x"
  for(i in 1:p) {
    try(aplot <- ValidPlotNew(fit = fit.stan, x = theDesign[,i], y=StanEmulator$tF, 
                              ObsRange = ObsRange, main = "", cex.main=0.8,
                              xlab=ParamNames[i], log=tlogs[i]), silent=TRUE)
    if(!inherits(aplot, "try-error") & !is.null(Obs)){
      abline(h=Obs, lty=2, col=4)
      if(is.null(ObsErr))
        warning("The observations do not have 0 error. Please add ObsErr else the plot will be misleading")
      else{
        abline(h=Obs+2*ObsErr, col=4, lty=2)
        abline(h=Obs-2*ObsErr, col=4, lty=2)
      }      }
  }
  return(fit.stan)
}

ValidationStan <- function(NewData, Emulator, main = ""){
#' Predict method for Gaussian Process model
#' 
#' @param NewData a data frame containing the inputs and outputs
#' @param Emulator a GP emulator from EMULATE.gpstan function
#' @param main a heading for the validation plots
#' 
#' @return A list contating 'Expectation' and 'Variance' together with the 
#' validation plots.
#'
  p <- dim(Emulator$Design)[2]
  if(p >= 2) Xpred <- NewData[,which((names(NewData)%in%colnames(Emulator$Design))|names(NewData)%in%Emulator$Factors)]
  else Xpred = NewData[, 1]
  emData <- EMULATOR.gpstan(Xpred, Emulator=Emulator)
  fit.stan <- cbind(emData$Expectation, emData$Expectation-2*sqrt(emData$Variance), 
                    emData$Expectation+2*sqrt(emData$Variance))
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
  else if(p<10){
    par(mfrow = c(3, 3), mar=c(4, 4, 1, 1))
  }
  else if(p<13){
    par(mfrow = c(4, 3), mar=c(4, 4, 1, 1))
  }
  else if(p<=16){
    par(mfrow = c(4, 4), mar=c(4, 4, 1, 1))
  }
  for(i in 1:p) {
    ValidPlot(fit.stan, X = Xpred, y=NewData[,which(names(NewData)==Emulator$ResponseString)], interval = range(fit.stan), axis = i, 
              heading = main, xrange = c(-1, 1), ParamNames = colnames(Emulator$Design)) 
  }
  fit.stan
}

##########################################################
#Design conversion functions
##########################################################
#All require specification of 
#param.lows (lower values of parameters)
#param.highs (max values of parameters)
#param.names (the names of the parameters)
#which.logs (pointers for those parameters that were designed on log scale)
#Mainly used in plotting

#Function to convert [-1,1] LHC to given scale
DesignConvert <- function(Xconts, param.names, param.lows, param.highs, which.logs=NULL){
  if(!(length(param.names)==length(param.lows)))
    stop("specify as many parameter names as parameter ranges")
  else if(!(length(param.highs)==length(param.lows)))
    stop("Mismatch in min and max values")
  conversion <- function(anX,lows,highs){
    ((anX+1)/2)*(highs-lows) +lows
  }
  param.lows.log <- param.lows
  param.highs.log <- param.highs
  param.lows.log[which.logs] <- log10(param.lows[which.logs])
  param.highs.log[which.logs] <- log10(param.highs[which.logs])
  tX <- sapply(1:length(param.lows), function(i) conversion(Xconts[,i],param.lows.log[i],param.highs.log[i]))
  tX[,which.logs] <- 10^tX[,which.logs]
  tX <- as.data.frame(tX)
  names(tX) <- param.names
  tX
}

#3. Function to convert [above scale to [-1, 1]
#Probably need to pass parameter ranges to be safe
DesignantiConvert <- function (Xconts){
  anticonversion <- function(newX,lows,highs){
    2*((newX-lows)/(highs-lows))-1
  }
  param.lows.log <- param.lows
  param.highs.log <- param.highs
  param.lows.log[which.logs] <- log10(param.lows[which.logs])
  param.highs.log[which.logs] <- log10(param.highs[which.logs])
  Xconts[,which.logs] <- log10(Xconts[,which.logs])
  tX <- sapply(1:length(param.lows), function(i) anticonversion(Xconts[,i],param.lows.log[i],param.highs.log[i]))
  tX <- as.data.frame(tX)
  names(tX) <- param.names
  tX
}
DesignantiConvert1D <- function (Xconts){
  anticonversion <- function(newX,lows,highs){
    2*((newX-lows)/(highs-lows))-1
  }
  param.lows.log <- param.lows
  param.highs.log <- param.highs
  param.lows.log[which.logs] <- log10(param.lows[which.logs])
  param.highs.log[which.logs] <- log10(param.highs[which.logs])
  Xconts[which.logs] <- log10(Xconts[which.logs])
  tX <- sapply(1:length(param.lows), function(i) anticonversion(Xconts[i],param.lows.log[i],param.highs.log[i]))
  tX <- as.data.frame(tX)
  tX
}

