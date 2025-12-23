#source('rotation_functions.R')

#' History matching
#'
#' Calculates the implausibility (for the \ell-dimensional field) for a sample of expectations and variances from basis emulators
#'
#' @param DataBasis object containing the basis used in emulation ($tBasis)
#' @param Obs observation vector (length \ell), must be centred
#' @param Expectation a matrix containing emulator expectations, where a given row contains the expectations for the q emulated basis vectors, for some x
#' @param Variance a matrix containing emulatorvariances, where a given row contains the variances for the q emulated basis vectors, for some x
#' @param Error observation error variance matrix
#' @param Disc discrepancy variance matrix
#' @param weightinv if not NULL, the inverse of W = var_err + var_disc, used for projection
#' #' 
#' @return \item{impl}{Vector of implausibilities corresponding to the rows of Expectation and Variance}
#' \item{bound}{The chi-squared bound for an \ell-dimensional field}
#' \item{nroy}{Proportion of parameter settings that are not ruled out, using bound}
#' \item{inNROY}{Vector indicating whether a parameter setting is ruled out}
#'
#' @export
HistoryMatch <- function(DataBasis, Obs, Expectation, Variance, Error, Disc, weightinv = NULL, BasisUncertainty = FALSE){
  q <- dim(Expectation)[2]
  Basis <- DataBasis$tBasis[,1:q]
  l <- dim(Basis)[1]
  stopifnot(q == dim(Variance)[2])
  W <- Error + Disc
  if (is.null(weightinv)){
    weightinv <- GetInverse(W)
  }
  R_W <- ReconError(Obs, Basis, weightinv = weightinv, scale = FALSE)
  # Project observations onto basis if required
  if (length(Obs) == l){
    ObsProj <- CalcScores(Obs, Basis, weightinv = weightinv)
  }
  # Add uncertainty from discarded basis vectors?
  if (BasisUncertainty == TRUE){
    BasisVar <- DiscardedBasisVariance(DataBasis, q, weightinv)
    W <- W + BasisVar
  }
  # Project variance matrices onto basis if required
  if (dim(Disc)[1] == l){
    WProj <- VarProj(W, Basis, weightinv = weightinv)
  }
  nn <- dim(Expectation)[1]
  impl <- as.numeric(mclapply(1:nn, function(i) ImplCoeff(Expectation[i,], Variance[i,], ObsProj, WProj, 0*WProj)))
  impl <- impl + rep(R_W, nn) 
  bound <- qchisq(0.995, l)
  nroy <- sum(impl < bound)/nn
  inNROY <- impl < bound
  return(list(impl = impl, bound = bound, nroy = nroy, inNROY = inNROY))
}

#' Coefficient implausibility
#'
#' Calculates the coefficient implausibility for a single x, given projected quantities
#'
#' @param Expectation length q vector with emulator expectations
#' @param Variance length q vector with emulator variances
#' @param Obs projected observations
#' @param Error projected observation error variance matrix
#' @param Disc projected discrepancy variance matrix
#'  
#' @return The coefficient implausibility (given the matrix used in projection)
#'
#' @export
ImplCoeff <- function(Expectation, Variance, Obs, Error, Disc){
  V <- Error + Disc + diag(Variance)
  Q <- chol(V)
  proj.output <- Expectation
  y <- backsolve(Q, as.vector(Obs - proj.output), transpose = TRUE)
  impl <- crossprod(y,y)
  return(impl)
}

#' Setting discrepancy multiple
#'
#' Scaling the discrepancy to ensure that the observations won't be ruled out
#'
#' @param Basis full basis
#' @param q where the basis is truncated
#' @param obs observations
#' @param level quantile of the chi-squared distribution to use (< 0.995)
#' @param weightinv inverse of W, to use in projection
#'  
#' @return scalar to be used as discrepancy multiplier, to ensure observations not ruled out
#'
#' @export
SetDiscrepancy <- function(tBasis, q, obs, level = 0.95, weightinv = NULL){
  TruncatedError <- ReconError(obs, tBasis[,1:q], weightinv = weightinv, scale = FALSE)
  l <- dim(tBasis)[1]
  b <- qchisq(level, l)
  DiscMultiplier <- c(TruncatedError / b)
  return(DiscMultiplier)
}


#' Adding discarded basis variance
#' 
#' Function that gives variance given by deleted basis vectors (Wilkinson 2010), used within HistoryMatch
#' 
#' @param DataBasis - object containing basis, centred field etc.
#' @param q - where the basis is truncated
#' @param weightinv inverse of W, to use in projection
#' 
#' @return Matrix containing uncertainty due to basis truncation
#' 
#' @export
DiscardedBasisVariance <- function(DataBasis, q, weightinv = NULL){
  BasMinusQ <- DataBasis$tBasis[,-(1:q)]
  DeletedCoeffs <- CalcScores(DataBasis$CentredField, BasMinusQ, weightinv)
  EstVar <- apply(DeletedCoeffs, 2, var)
  DeletedBasisVar <- BasMinusQ %*% diag(EstVar) %*% t(BasMinusQ)
  return(DeletedBasisVar)
}
