twd <- getwd()
packages <- c('GenSA', 'far', 'fields', 'lhs', 'maps', 'mco', 'mvtnorm', 
              'ncdf4', 'parallel', 'rstan', 'shape', 'tensor', 'withr', 'tgp', 'CGP',
              'loo', 'bayesplot', 'gridExtra')
sapply(packages, require, character.only = TRUE)

source("BuildEmulator/AutoLMcode.R")
source("BuildEmulator/CustomPredict.R")
source("BuildEmulator/JamesNewDevelopment.R")
source("BuildEmulator/DannyDevelopment.R")

# Stan files for Nonstationary GP Emulator in Stan with fixed nugget
tfile_loc_nst = paste(twd, "/BuildEmulator/FitNGP.stan", sep = "")
tprednewfile_loc_nst = paste(twd, "/BuildEmulator/PredictGenNGP.stan", sep = "")

# Stan files to estimate mixture components
tfile_mix = paste(twd, "/BuildEmulator/MixtureModFit.stan", sep = "")

ccode_fit_nst <- stanc(file = tfile_loc_nst)
model_fit_nst <- stan_model(stanc_ret = ccode_fit_nst)


ccode_predict_nst <- stanc(file = tprednewfile_loc_nst) # predictions could be perfectly outside the Stan.
model_predict_nst <- stan_model(stanc_ret = ccode_predict_nst) # predictions could be perfectly outside the Stan. 

ccode_fit_mix <- stanc(file = tfile_mix)
model_fit_mix <- stan_model(stanc_ret = ccode_fit_mix)

gpstan.default.params <- list(SigSq = 0.15, SigSqV = 0.25, AlphaAct = 4, BetaAct = 4,
                              AlphaInact = 42, BetaInact = 9, AlphaNugget = 3,
                              BetaNugget = 0.1, AlphaRegress = 0, BetaRegress = 10, 
                              nugget = 0.0001, SwitchDelta = 1, SwitchNugget = 1, SwitchSigma = 1)


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
  if(maxdf < 2)
    return(list(linModel=startLM,Names=NULL,mainEffects=NULL,Interactions=NULL,Factors=NULL,FactorInteractions=NULL,ThreeWayInters=NULL,DataString=dString,ResponseString=Response,tData=tData,BestFourier=TRUE))
  if(TryFouriers)
    msl <- list(linModel=startLM,Names=NULL,mainEffects=NULL,Interactions=NULL,Factors=NULL,FactorInteractions=NULL,ThreeWayInters=NULL,DataString=dString,ResponseString=Response,tData=tData,BestFourier=TRUE,maxOrder=maxOrder)
  else
    msl <- list(linModel=startLM,Names=NULL,mainEffects=NULL,Interactions=NULL,Factors=NULL,FactorInteractions=NULL,ThreeWayInters=NULL,DataString=dString,ResponseString=Response,tData=tData,BestFourier=FALSE)
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

MIXTURE.design <- function(formula, tData.mixture, L=2, 
                           CompiledModel = model_fit_mix) {
#' Generates a matrix of mixture components for X
#' 
#'  @param formula an object of class `formula`: a symbolic description of 
#'  the model to be fitted.
#'  @param tData.mixture a data frame, containing the variables in the model
#'  and standard errors from stationary fit.
#'  @param L an integer, a number of mixtures 
#'  @param CompiledModel an instance of S4 class stanmodel used for fitting finite mixture model 
#'  
#'  @return A list contatining the following components (need to write up this later)
#'  
  
  m <- model.frame(formula, tData.mixture)
  H <- model.matrix(formula, tData.mixture)
  std.err <- tData.mixture$std.err
  BayesMixture <- sampling(CompiledModel, data = list(N = length(std.err), D = dim(H)[2], 
                                                      K = L, y = std.err, x = H), 
                           iter = 10000, warmup=5000, chains = 2, cores = 4)
  MixtureSamples <- extract(BayesMixture, pars = c('beta', 'sigma'))
  # extract only the posterior mean for mixture matrix for design set
  MixtureMat <- summary(BayesMixture, pars = 'mixture_vec')$summary[, 1]
  MixtureMat <- t(matrix(MixtureMat, nrow = L))
  
  mixture.list <- list(std.err = std.err, X = H, L = L, 
                       MixtureSamples=MixtureSamples,
                       MixtureMat = MixtureMat, StanObject = BayesMixture, 
                       formula = formula)
  return(mixture.list)
}
SoftMax <- function(beta, x) {
  m <- exp(beta%*%t(x))
  return(t(m%*%diag(1/colSums(m))))
}

MIXTURE.predict <- function(x, mixtureComp, FastVersion = FALSE, batches = 500) { 
#' Generates a matrix of mixture components for unseen x
#' 
#' @param x a matrix of inputs
#' @param mixtureComp a mixture model from MIXTURE.design function
#' @param CompiledModel an instance of S4 class stanmodel used for fitting a GP model
#' @param FastVersion if TRUE compute mixture components in parallel
#' @param batches an integer defining a number inside batches.
#' 
#' @return A matrix of mixture components for the validation matrix x
  tt = terms(mixtureComp$formula)
  Terms = delete.response(tt)
  mm = model.frame(Terms, x)
  H = model.matrix(Terms, mm)
  if(FastVersion) {
    beta = mixtureComp$MixtureSamples$beta[5001:10000, , ]
    Nruns = dim(H)[1]
    Nbatch = ceiling(Nruns/batches)
    batchSeq = lapply(1:(Nbatch-1), function(k) seq(from=1+(k - 1)*batches,by=1,length.out = batches))
    batchSeq[[Nbatch]] = (batchSeq[[Nbatch-1]][batches]+1):Nruns
    # maybe I have to do DoClusters
    cl = makeCluster(2)
    registerDoParallel(cl)
    predict.mixture.mean <- foreach(i=1:length(batchSeq), .combine = rbind) %dopar% {
      N = dim(beta)[1]
      if(dim(H)[2] == 1) {
        H_batch = matrix(H[batchSeq[[i]], ], ncol = 1)
        tSamples = lapply(1:dim(beta)[1], function(e) exp(as.matrix(H_batch)%*%t(beta[e, ])))
      } else {
        H_batch = H[batchSeq[[i]], ]
        tSamples = lapply(1:dim(beta)[1], function(e) exp(as.matrix(H_batch)%*%t(beta[e, , ])))
      }
      tSamples = lapply(1:dim(beta)[1], function(e) tSamples[[e]]/rowSums(tSamples[[e]]))
      predict.mixture.mean.samples = Reduce("+", tSamples)
      (1/N)*predict.mixture.mean.samples
    }
    stopCluster(cl = cl)
  }
  else {
    Nsamples <- dim(mixtureComp$MixtureSamples$sigma)[1]
    tSamples <- mclapply(1:Nsamples, function(e) exp(as.matrix(H)%*%t(mixtureComp$MixtureSamples$beta[e, ,])),
                         mc.cores = 6)
    tSamples <- mclapply(1:Nsamples, function(e) tSamples[[e]]/rowSums(tSamples[[e]]), mc.cores = 6)
    predict.mixture.mean <- Reduce("+", tSamples)
    predict.mixture.mean <- (1/Nsamples)*predict.mixture.mean
  }
  return(predict.mixture.mean)
}

EMULATE.gpstanNSt <- function(meanResponse, CompiledModelFit = model_fit_nst, sigmaPrior = FALSE, nuggetPrior=FALSE, 
                              activePrior = FALSE, activeVariables = NULL, tData, additionalVariables=NULL, FastVersion = FALSE, 
                              mixtureComp, prior.params = gpstan.default.params, ...) {
#' Function implement Nonstationary Gaussian process (GP) emulator 
#' 
#' @param meanResponse the output of EMULATE.lm containing a linear model emulator
#' @param CompiledModelFit an instance of S4 class stanmodel used for fitting a GP model
#' @param sigmaPrior a logical argument with FALSE (default) specifying sigma prior parameters
#' at the values found from the linear model fit.
#' @param nuggetPrior a logical argument with FALSE (default) specifying nugget 
#' parameter at fixed value. TRUE specifying a prior distribution for the nugget parameter
#' @param activePrior a logical argument with FALSE (default) specifying the same prior
#' for correlation length parameters. TRUE specifying different priors for correlation length
#' parameters.
#' @param activeVariables is a vector of parameter names that we want to fit a GP to that didn't make it intp the model
#' @param tData a data frame of inputs X and a vector of output responses Z
#' @param additionalVaribles a vector of characters corresponding to the names 
#' of additional variables. NULL (default)
#' @param CreatePredictObject TRUE value results in generating emulator predictions for input matrix
#' @param FastVersion TRUE value results at saving parameter values
#'  at the posterior mean; FALSE (default) saves the posterior samples for parameters
#' @param mixtureComp a mixture model from MIXTURE.design function
#' @param prior.params a list of parameters to the prior specification. The default is 
#' gpstan.default.params.
#' 
#' @return A GP emulator object.

  Design <- as.matrix(tData[,which((names(tData)%in%meanResponse$Names) |  (names(tData)%in%additionalVariables) | (names(tData) %in% names(meanResponse$Fouriers)))])
  tF <- tData[,which(names(tData)==meanResponse$ResponseString)]
  H <- model.matrix(meanResponse$linModel)
  N1 <- dim(Design)[1]
  Np <- dim(H)[2]
  # define the number of clusters and the transpose of a mixture matrix
  L <- mixtureComp$L
  B <- t(mixtureComp$MixtureMat)
  if(sigmaPrior) prior.params$SwitchSigma = 2
  if(nuggetPrior) {
    prior.params$SwitchNugget = 2
    prior.params$UpperLimitNugget = prior.params$BetaNugget/(prior.params$AlphaNugget-1) + 10*sqrt(prior.params$BetaNugget^2/((prior.params$AlphaNugget-1)^2*(prior.params$AlphaNugget-2)))
  }
  else {
    prior.params$UpperLimitNugget = 2*prior.params$nugget
  }
  if(activePrior) prior.params$SwitchDelta = 2
  if(!sigmaPrior) {
    consEm <- EMULATE.lm(Response=meanResponse$ResponseString, tData=tData, tcands="Noise",tcanfacs=NULL,TryFouriers=TRUE,maxOrder=2,maxdf = 0)
    sd2 <- summary(consEm$linModel)$sig - summary(meanResponse$linModel)$sig
    sigsq <- summary(meanResponse$linModel)$sigma
    sigsqvar <- sd2
    # consider the constant Gaussian Process mean
    if(Np == 1) sigsqvar <- summary(meanResponse$linModel)$sig
    prior.params$SigSq <- sigsq
    prior.params$SigSqV <-sigsqvar
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
    init.list <- list(list(beta=array(meanResponse$linModel$coefficients, dim = Np), sigma = array(rep(prior.params$SigSq, L), dim = L),
                           delta_par=matrix(c(rep(0.05, L*p.active), rep(0.7, L*p.inactive)), nrow = L, ncol = (p.active + p.inactive)), 
                           nugget = array(rep(prior.params$nugget, L), dim = L)),
                      list(beta=array(meanResponse$linModel$coefficients, dim = Np), sigma=array(rep(prior.params$SigSq, L), dim = L),
                           delta_par=matrix(c(rep(0.1, L*p.active), rep(1, L*p.inactive)), nrow = L, ncol = (p.active + p.inactive)), 
                           nugget = array(rep(prior.params$nugget, L), dim = L)))
    StanEmulator <- sampling(CompiledModelFit, data = list(N1 = N1, pact = p.active, pinact = p.inactive, 
                                                           p = p, Np = Np, SwitchDelta = prior.params$SwitchDelta, 
                                                           SwitchNugget = prior.params$SwitchNugget, SwitchSigma = prior.params$SwitchSigma, 
                                                           SigSq = prior.params$SigSq, SigSqV = prior.params$SigSqV, 
                                                           AlphaAct = prior.params$AlphaAct, BetaAct = prior.params$BetaAct, 
                                                           AlphaInact = prior.params$AlphaInact, BetaInact = prior.params$BetaInact, 
                                                           AlphaNugget = prior.params$AlphaNugget, BetaNugget = prior.params$BetaNugget, 
                                                           AlphaRegress = prior.params$AlphaRegress, BetaRegress = prior.params$BetaRegress, 
                                                           nuggetfix = prior.params$nugget, 
                                                           UpperLimitNugget = prior.params$UpperLimitNugget,
                                                           X1 = Design, y1= tF, H1 = H, L = L, A = B), 
                             iter = 2000,warmup=1000, chains = 2, cores = 2, init=init.list,
                             pars = c('nugget', 'sigma', 'delta_par', 'beta', 'log_lik'), ...)
  } else {
    # consider the same prior specification for delta_par (correlation length parameters)
    Design <- as.matrix(tData[,which((names(tData)%in%meanResponse$Names) |  (names(tData)%in%additionalVariables) | (names(tData) %in% names(meanResponse$Fouriers)))])
    p <- dim(Design)[2]
    p.active = p.inactive = 1
    init.list <- list(list(beta=array(meanResponse$linModel$coefficients, dim = Np), sigma = array(rep(prior.params$SigSq, L), dim = L),
                           delta_par=matrix(rep(0.7, L*p), nrow = L, ncol = p), nugget = array(rep(prior.params$nugget, L), dim = L)),
                      list(beta=array(meanResponse$linModel$coefficients, dim = Np), sigma=array(rep(prior.params$SigSq, L), dim = L),
                           delta_par=matrix(rep(1, L*p), nrow = L, ncol = p), nugget = array(rep(prior.params$nugget, L), dim = L)))
    StanEmulator <- sampling(CompiledModelFit, data = list(N1 = N1, pact = p.active, pinact = p.inactive, 
                                                           p = p, Np = Np, SwitchDelta = prior.params$SwitchDelta, 
                                                           SwitchNugget = prior.params$SwitchNugget, SwitchSigma = prior.params$SwitchSigma, 
                                                           SigSq = prior.params$SigSq, SigSqV = prior.params$SigSqV, 
                                                           AlphaAct = prior.params$AlphaAct, BetaAct = prior.params$BetaAct, 
                                                           AlphaInact = prior.params$AlphaInact, BetaInact = prior.params$BetaInact, 
                                                           AlphaNugget = prior.params$AlphaNugget, BetaNugget = prior.params$BetaNugget, 
                                                           AlphaRegress = prior.params$AlphaRegress, BetaRegress = prior.params$BetaRegress, 
                                                           nuggetfix = prior.params$nugget, 
                                                           UpperLimitNugget = prior.params$UpperLimitNugget,
                                                           X1 = Design, y1= tF, H1 = H, L = L, A = B), 
                             iter = 2000,warmup=1000, chains = 2, cores = 2, init=init.list,
                             pars = c('nugget', 'sigma', 'delta_par', 'beta', 'log_lik'),...)
    
  }
  ParameterSamples <- extract(StanEmulator, pars = c('sigma', 'delta_par', 'beta', 'nugget'))
  if(FastVersion){
    lps <- extract_log_lik(StanEmulator)
    tMAP <- which.max(rowSums(lps))
    A <- diag(ParameterSamples$nugget[tMAP, apply(mixtureComp$MixtureMat, 1, which.max)], ncol = N1, nrow = N1)
    for(l in 1:L) {
      A <- A + ParameterSamples$sigma[tMAP, l]^2*diag(B[l, ])%*%CovMatrix(Design, ParameterSamples$delta_par[tMAP, l, ])%*%diag(B[l, ])
    }
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
  return(c(meanResponse, mixtureComp, gp.list))
}
ValidPlot <- function(fit, X, y, interval = c(), axis = 1, heading = " ", xrange = c(-1, 1), ParamNames) {
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
         xlab=ParamNames[axis], ylim=c(minval, maxval),
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



virtual.LOO.NSt <- function(Design, y, cls, sigma, H, beta, nugget, 
                            MixtureMat, L) {
  #' Function to compute the virtual Leave-One-Out formulas. 
  #' 
  #' @param Design a matrix of inputs for design set.
  #' @param y a vector of simulator responses at Design.
  #' @param cls a matrix of correlation length parameters. 
  #' Each row corresponds to correlation length parameters from an individual input region.
  #' @param sigma a vector of scale parameters.
  #' @param H a matrix of regression functions evaluted at the design points.
  #' @param beta a vector of regression coefficients.
  #' @param nugget a vector of a nugget parameters. 
  #' @param MixtureMat a transpose of a matrix with mixture values.
  #' @param L a number of input regions.
  #'  
  #' @return  a random generation for the normal distribution with mean and standard deviation
  #' found using the virtual Leave-One-Out formulas.

  expectation <- c()
  variance <- c()
  predict_y <- c()
  Gamma2 <- diag(nugget[apply(MixtureMat, 2, which.max)], ncol = dim(Design)[1],
                 nrow = dim(Design)[1])
  for(l in 1:L) Gamma2 <- Gamma2 + sigma[l]^2*diag(MixtureMat[l, ])%*%CovMatrix(Design, cls[l, ])%*%diag(MixtureMat[l, ])
  #for(l in 1:L) Gamma2 <- Gamma2 + sigma[l]^2*diag(MixtureMat[l, ])%*%CovMatrix(Design, cls[l])%*%diag(MixtureMat[l, ])
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

LOO.plot.NSt <- function(StanEmulator, ParamNames) {
  #' Function to generate Leave-One-Out predictions
  #' 
  #'   @param StanEmulator a Nonstationary GP Emulator from EMULATE.gpstanNSt
  #'   @param ParamNames a vector of names for parameters
  #'   
  #'   @return a data frame with three columns, with first column corresponding to posterior mean
  #'   and second and third columns corresponding to the minus and plus two standard deviations.
  #'   The function also generates LOO validation plots.
  
  MixtureMat <- t(StanEmulator$MixtureMat) 
  predict.y <- sapply(1:dim(StanEmulator$ParameterSamples$delta_par)[1], function(k)
    virtual.LOO.NSt(Design = StanEmulator$Design, y = StanEmulator$tF,
                    H = StanEmulator$H, cls = StanEmulator$ParameterSamples$delta_par[k, ,],
                    sigma = StanEmulator$ParameterSamples$sigma[k, ],
                    beta = StanEmulator$ParameterSamples$beta[k, ], 
                    nugget = StanEmulator$ParameterSamples$nugget[k, ], 
                    MixtureMat = MixtureMat, L = StanEmulator$L))
  mean.predict.y <- rowMeans(predict.y)
  sd.predict.y <- apply(predict.y, 1, sd)
  fit.stan <- data.frame(cbind(mean.predict.y, mean.predict.y-2*sd.predict.y, 
                               mean.predict.y + 2*sd.predict.y))
  names(fit.stan) <- c('posterior mean', 'lower quantile', 'upper quantile')
  p <- dim(StanEmulator$Design)[2]
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
  for(i in 1:p) {
    ValidPlot(fit.stan, StanEmulator$Design, StanEmulator$tF, interval = range(fit.stan), axis = i, 
              heading = "", xrange = c(-1, 1), ParamNames = ParamNames) 
  }
  return(fit.stan)
}


EMULATOR.gpstanNSt <- function(x, Emulator, mixtureComp, GP=TRUE, FastVersion=FALSE, CompiledModelPredict = model_predict_nst){
  #' Predict method for Nonstationary Gaussian Process models
  #' 
  #' @param x a matrix of inputs 
  #' @param Emulator a GP emulator from EMULATE.gpstan.NSt function
  #' @param mixtureComp a mixture model from MIXTURE.design function
  #' @param GP not sure about this (?)
  #' @param FastVersion TRUE value results at saving parameter values
  #'  at the posterior mean; FALSE (default) saves the posterior samples for parameters
  #' @param CompiledModelPredict an instance of S4 class stanmodel that is used to produe predictions
  #' 
  #' @return A list contatining 'Expectation' and 'Variance'
  XX <- x # regression for the mixture model
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
      Xpred <- Xpred[,tinds] 
    } else {
      Xpred <- matrix(Xpred[, 1], ncol = 1)
    }
    L <- Emulator$L
    A1 <- t(Emulator$MixtureMat)
    if(FastVersion){
      if(is.null(Emulator$FastParts))
        stop("Build a FastVersion with EMULATE.gpstan first")
      A2 <- t(MIXTURE.predict(XX, mixtureComp, FastVersion = TRUE))
      Covtx <- Emulator$ParameterSamples$sigma[Emulator$FastParts$tMAP, 1]^2*diag(A1[1,])%*%NewCov(Xpred, Design=Emulator$Design, cls = Emulator$ParameterSamples$delta[Emulator$FastParts$tMAP, 1, ])%*%diag(A2[1, ])
      for(l in 2:L) Covtx <- Covtx + Emulator$ParameterSamples$sigma[Emulator$FastParts$tMAP, l]^2*diag(A1[l, ])%*%NewCov(Xpred, Design=Emulator$Design, cls = Emulator$ParameterSamples$delta[Emulator$FastParts$tMAP, l, ])%*%diag(A2[l, ])
      txtA <- backsolve(Emulator$FastParts$QA,Covtx,transpose=TRUE)
      tExpectation <- as.vector(Hpred%*%Emulator$ParameterSamples$beta[Emulator$FastParts$tMAP,] + crossprod(txtA, Emulator$FastParts$Ldiff))
      tVariance <- as.vector(diag(diag(as.vector(matrix(Emulator$ParameterSamples$sigma[Emulator$FastParts$tMAP, ]^2, ncol=L)%*%A2^2), nrow=dim(Xpred)[1],ncol=dim(Xpred)[1]) - crossprod(txtA)))
      StandardDev <- sqrt(tVariance)
    }
    else{ 
      A2 <- t(MIXTURE.predict(XX, mixtureComp, FastVersion = FALSE))
      fit.y2 <- sampling(CompiledModelPredict, 
                         data = list(N1 = dim(Emulator$Design)[1], N2 = dim(Xpred)[1], 
                                     p = dim(Emulator$Design)[2], M = dim(Emulator$ParameterSamples$beta)[1], 
                                     Np = dim(Emulator$H)[2], L = L, X1 = Emulator$Design, X2 = Xpred, y1 = Emulator$tF, 
                                     H1 = Emulator$H, H2 = Hpred, A1 = A1, A2 = A2, beta = Emulator$ParameterSamples$beta, 
                                     delta = Emulator$ParameterSamples$delta, sigma = Emulator$ParameterSamples$sigma, 
                                     nugget = Emulator$ParameterSamples$nugget), 
                         iter = 1, warmup = 0, chains = 1, cores = 1, pars = c("tmeans", "tsds"), include = TRUE, 
                         algorithm = c('Fixed_param'))
      predict.y2 <- extract(fit.y2, pars = c('tmeans','tsds'))
      tExpectation <- predict.y2$tmeans[1,]
      StandardDev <- predict.y2$tsds[1,]
    }
    return(list(Expectation=tExpectation,Variance=StandardDev^2))
  }
}

EMULATOR.gpstanNSt.multicore <- function(x, Emulator, mixtureComp, GP=TRUE, FastVersion=FALSE, batches=500){
  #' Predict method for Gaussian process model for many runs
  #' 
  #' @param x a matrix of  
  #' @param Emulator a GP emulator from EMULATE.gpstan function
  #' @param mixtureComp a mixture model from MIXTURE.design function
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
  tSamples <- mclapply(batchSeq, function(e) EMULATOR.gpstanNSt(x=x[e,], Emulator=Emulator, mixtureComp=mixtureComp,
                                                                GP=GP, FastVersion = FastVersion), mc.cores = 6)
  tExpectation <- rep(NA,Nruns)
  tVariance <- rep(NA,Nruns)
  for(i in 1:Nbatch){
    tExpectation[batchSeq[[i]]] <- tSamples[[i]]$Expectation
    tVariance[batchSeq[[i]]] <- tSamples[[i]]$Variance
  }
  return(list(Expectation=tExpectation,Variance=tVariance))
}



ValidationStanNSt <- function(NewData, Emulator, mixtureComp, main = ""){
#' Predictions Nonstationary Gaussian Process models for validation set
#' 
#' @param NewData a data frame of inputs and response variable
#' @param Emulator a GP emulator from EMULATE.gpstan.NSt function
#' @param mixturePredict a mixture matrix for x
#' 
#' @return A list contatining 'Expectation' and 'Variance'.
#' Validation plots against X. The predictions and two standard deviation
#' prediction intervals for unseen points are in black. The true values are
#' in either green, if they are within two standard deviations of the 
#' prediction, or red otherwise.
  
  p <- dim(Emulator$Design)[2]
  if(p >= 2) Xpred <- NewData[,which((names(NewData)%in%colnames(Emulator$Design))|(names(NewData)%in%Emulator$Factors))]
  else Xpred = NewData[, 1]
  emData <- EMULATOR.gpstanNSt(Xpred, Emulator=Emulator, mixtureComp = mixtureComp)
  fit.stan <- cbind(emData$Expectation, emData$Expectation-2*sqrt(emData$Variance), 
                    emData$Expectation+2*sqrt(emData$Variance))
  p <- dim(Emulator$Design)[2]
  if(p<2){
    par(mfrow = c(1, 1), mar=c(4, 4, 1, 1))
  }
  else if(p <=4){
    par(mfrow = c(2, 2), mar=c(4, 4, 1, 1))
  }
  else if(p<=9){
    par(mfrow = c(3, 3), mar=c(4, 4, 1, 1))
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

CalStError <- function(LOOs, true.y) {
#' Calculate the Standardized error from GP emulator
#' 
#'   @param LOOs data frame of LOOs from stanLOOplotNSt
#'   @param true.y a vector of simulator evaluations
#'   
#'   @return a vector of standardized errors
#'   
  st.error <- (true.y - LOOs[, 1])/(0.5*(LOOs[, 1]-LOOs[, 2]))
  return(st.error)
}
RMSE.Fun <- function(pred, y.true) {
#' Calculate the Root Mean Squared Error (RMSE) diagnostics.
#' 
#' @param pred a list containing 'Expectation' and 'Variance'.
#' Usually an output from functions 'EMULATOR.gpstan', 'EMULATOR.gpstanNSt'
#' 'ValidationStan' and 'ValidationStanNSt'
#' @param y.true a vector of simulator evaluations for validation set
#' 
#' @return The RMSE value
#' 
  N <- length(y.true)
  result <- sqrt(sum((pred-y.true)^2)/N)
  return(result)
}
S.Int.score <- function(pred, y.true, alpha = 0.05) {
#' Function to calculate the Interval Score (IS) for the central 
#' (1-alpha)x100 with lower and upper endpoints at the predictive quantiles 
#' alpha/2 and 1-alpha/2
#' 
#' @param pred a data frame with the first column corresponding to posterior mean
#' second and thrid columns to two standard deviation prediction intervals
#' @param y.true a vector of simulator evaluations for validation set
#' @param alpha level of desired prediction
#' 
#' @return The Interval Score (IS)
  score <- c()
  u <- pred[, 3]
  l <- pred[, 2]
  N <- length(y.true)
  for(i in 1:N) {
    score[i] <- (u[i]-l[i]) + 2/alpha*(l[i]-y.true[i])*ifelse(y.true[i]<l[i], 1, 0) + 2/alpha*(y.true[i]-u[i])*ifelse(y.true[i]>u[i], 1, 0)
  }
  result <- sum(score)/N
  return(result)
}
VAIC <- function(mixtureComp, nuggetParam = FALSE) {
  #' Function to calculate the modified AIC criterion: the first part of
  #' the modified AIC criterion corresponds to the deviance (twice the negative
  #' log likelihood), the second term is equal to twice the number of parameters 
  #' in the mixture model. We also add the penalty that corresponds to the number
  #' of region-specific parameters of full nonstationary GP model.
  #' @param mixtureComp 
  #' @nuggetParam logical. If FALSE (the default) the nugget parameter is fixed at a
  #' small arbitrary value for numerical stability of computations.

  #' @return a real number. Please note that we choose L with the lowest AIC score.
  if(dim(mixtureComp$X)[2] == 1) {
    post.mean.beta = t(matrix(sapply(1:mixtureComp$L, function(x) mean(mixtureComp$MixtureSamples$beta[, x, ])), ncol = mixtureComp$L))
  }else {
    post.mean.beta = t(sapply(1:mixtureComp$L, function(x) colMeans(mixtureComp$MixtureSamples$beta[, x, ])))
  }
  post.mean.sigma = colMeans(mixtureComp$MixtureSamples$sigma)
  mixture.comp = exp(as.matrix(mixtureComp$X)%*%t(post.mean.beta))
  mixture.comp = mixture.comp/rowSums(mixture.comp)
  std.err.L = sapply(1:mixtureComp$L, function(x) dnorm(mixtureComp$y.std, 0, post.mean.sigma[x]))
  
  Likelihood = prod(rowSums(sapply(1:mixtureComp$L, function(x) mixture.comp[, x]*std.err.L[, x])))
  loglike = log(Likelihood)
  Deviance  = -2*loglike
  L = mixtureComp$L
  p = dim(mixtureComp$X)[2]
  if(nuggetParam) {
    free.param =  L*((p+1)+(p+2))
    VAICscore = Deviance + 2*L*(p+1) + L*(p+2)
  } else {
    free.param =  L*((p+1)+(p+1))
    VAICscore = Deviance + 2*L*(p+1) + L*(p+1)
  }
  return(list(Deviance  = Deviance, VAICscore = VAICscore, 
              free.param = free.param))
}





