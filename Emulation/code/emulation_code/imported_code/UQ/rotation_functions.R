#' Formatting data
#'
#' Formats data so that it is in the correct form for use in other functions, and calculates the (weighted) SVD basis of the ensemble
#'
#' @param data a matrix containing individual fields in the columns (i.e. the matrix has dimension lxn)
#' @param weightinv the inverse of lxl positive definite weight matrix W. If NULL, the identity matrix is used
#' @param RemoveMean if TRUE, centres the data prior to calculating the basis
#' @param StoreEigen if TRUE, stores Q, lambda from eigendecomposition of W (in order to make later calculations more efficient)
#'
#' @return \item{tBasis}{The (weighted) SVD basis of the centred ensemble if RemoveMean = TRUE, of the original data otherwise}
#' \item{CentredField}{The centred data if RemoveMean = TRUE, the original data otherwise.}
#' \item{EnsembleMean}{The mean across the columns of the data. A zero vector if RemoveMean = FALSE}
#' \item{}
#'
#' @export
MakeDataBasis <- function(data, weightinv = NULL, W = NULL, RemoveMean = TRUE, StoreEigen = TRUE){
  if (RemoveMean == TRUE){
    EnsembleMean <- apply(data, 1, mean)
    CentredField <- 0*data
    for (i in 1:dim(data)[2]){
      CentredField[,i] <- data[,i] - EnsembleMean
    }
  }
  else {
    EnsembleMean <- c(rep(0, dim(data)[1]))
    CentredField <- data
  }
  #if (is.null(weightinv)){
  #  weightinv <- diag(dim(data)[1])
  #}
  if (is.null(W)){
    tSVD <- wsvd(t(CentredField), weightinv = weightinv)
    tBasis <- tSVD$v
    if (StoreEigen == TRUE){
      Q <- tSVD$Q
      Lambda <- tSVD$Lambda
      return(list(tBasis = tBasis, CentredField = CentredField, EnsembleMean = EnsembleMean, Q = Q, Lambda = Lambda))
    }
    else {
      return(list(tBasis = tBasis, CentredField = CentredField, EnsembleMean = EnsembleMean))
    }
  }
  else if (!is.null(W) & is.null(weightinv)){
    eig <- eigen(W)
    Q <- eig$vectors
    Lambda <- 1 / eig$values
    Winv <- Q %*% diag(Lambda) %*% t(Q)
    attr(Winv, 'diagonal') <- FALSE
    attr(Winv, 'identity') <- FALSE
    tSVD <- wsvd(t(CentredField), weightinv = Winv, Q = Q, Lambda = Lambda)
    tBasis <- tSVD$v
    if (StoreEigen == TRUE){
      return(list(tBasis = tBasis, CentredField = CentredField, EnsembleMean = EnsembleMean, Q = Q, Lambda = Lambda, Winv = Winv))
    }
    else {
      return(list(tBasis = tBasis, CentredField = CentredField, EnsembleMean = EnsembleMean, Winv = Winv))
    }
  }
}


#' Weighted singular value decomposition
#'
#' Calculates the SVD basis across the output, given the inverse of W.
#'
#' @param data n x l matrix to calculate basis from (i.e. rows are output fields).
#' @param weightinv l x l inverse of W. If NULL, calculates standard SVD.
#' @param Q l x l matrix from eigen decomposition of W^{-1}, if already have this then speeds up calculation of basis
#' @param Lambda vector from eigen decomposition of W^{-1}, if already have this then speeds up calculation of basis
#'
#' @return The weighted SVD of the data.
#'
wsvd <- function(data, weightinv = NULL, Q = NULL, Lambda = NULL){
  if (is.null(weightinv)){
    svd_output <- svd(data)
  }
  else {
    stopifnot(dim(data)[2] == dim(weightinv)[1])
    if (is.null(Q) & attributes(weightinv)$diagonal == FALSE){
      eig <- eigen(weightinv)
      Q <- eig$vectors
      Lambda <- eig$values
      data_w <- data %*% Q %*% diag(sqrt(Lambda)) %*% t(Q)
      svd_output <- svd(data_w)
      svd_output$v <- t(t(svd_output$v) %*% Q %*% diag(1 / sqrt(Lambda)) %*% t(Q))
      svd_output$Q <- Q
      svd_output$Lambda <- Lambda
    }
    else if (is.null(Q) & attributes(weightinv)$diagonal == TRUE){
      diag_values <- diag(weightinv)
      data_w <- data %*% diag(sqrt(diag_values))
      svd_output <- svd(data_w)
      svd_output$v <- t(t(svd_output$v) %*% diag(1 / sqrt(diag_values)))
    }
    else if (!is.null(Q)){
      data_w <- data %*% Q %*% diag(sqrt(Lambda)) %*% t(Q)
      svd_output <- svd(data_w)
      svd_output$v <- t(t(svd_output$v) %*% Q %*% diag(1 / sqrt(Lambda)) %*% t(Q))
      svd_output$Q <- Q
      svd_output$Lambda <- Lambda
    }
  }
  return(svd_output)
}

#' Matrix inversion via cholesky decomposition
#'
#' Inverts matrix W, assigning attributes for whether W is diagonal, to speed up other calculations.
#'
#' @param W square positive definite variance matrix
#'
#' @return Inverse of W, with attributes 'identity' and 'diagonal', used by other functions in the package to make calculations more efficient.
#'
#' @examples Winv <- GetInverse(diag(100))
#' attributes(Winv) # diagonal = TRUE, identity = TRUE
#'
#' Winv2 <- GetInverse(runif(100,0.1,1)*diag(100))
#' attributes(Winv2) # diagonal = TRUE, identity = FALSE
#'
#' Winv3 <- GetInverse(seq(0.1,1,length=100) %*% t(seq(0.1,1,length=100)) + 0.1*diag(100))
#' attributes(Winv3) # diagonal = FALSE, identity = FALSE
#'
#' @export
GetInverse <- function(W){
  diagmat <- all(W[lower.tri(W)] == 0, W[upper.tri(W)] == 0)
  if (diagmat == TRUE){
    InvW <- diag(1 / diag(W))
  }
  else {
    Q <- chol(W)
    y <- backsolve(Q, diag(dim(W)[1]), transpose = TRUE)
    InvW <- crossprod(y, y)
  }
  attr(InvW, 'diagonal') <- diagmat
  if (all(diag(InvW) == 1) & diagmat == TRUE){
    attr(InvW, 'identity') <- TRUE
  }
  else {
    attr(InvW, 'identity') <- FALSE
  }
  return(InvW)
}





#' Field reconstructions from coefficients
#'
#' Given a vector of coefficients for a basis, calculates the field
#'
#' @param coeffs Coefficient vector
#' @param basis Basis matrix
#' @return Reconstructed field.
#'
#' @export
Recon <- function(coeffs, basis){
  if (is.null(dim(basis)[2])){
    q <- 1
  }
  else {
    q <- dim(basis)[2]
  }
  stopifnot(length(coeffs) == q)
  if (is.null(dim(basis)[2])){
    reconstruction <- basis*as.numeric(coeffs)
  }
  else {
    reconstruction <- basis%*%as.numeric(coeffs)
  }
  return(reconstruction)
}

#' Calculate a set of vectors from weights and a basis
#'
#' Give basis vectors from linear combinations
#'
#' @param weights A vector of weights
#' @param basis Original basis
#'
#' @return New basis vector(s)
#'
#' @export
ReconBasis <- function(weights, basis){
  n <- dim(basis)[2]
  q <- length(weights) / n
  if (q == 1){
    new.basis <- as.vector(tensor(basis, weights, 2, 1))
  }
  else {
    dim(weights) <- c(n, q)
    new.basis <- tensor(basis, weights, 2, 1)
  }
  return(new.basis)
}



#' Project and reconstruct a given field
#'
#' Gives the reconstruction of a field using a basis, by projecting and back-projecting on this basis.
#'
#' @param obs Vector over original field
#' @param basis Basis matrix
#'
#' @return Reconstruction of the original field.
#'
#' @examples
#'
#' @export
ReconObs <- function(obs, basis, ...){
  nb <- is.null(dim(basis))
  if(!nb)
    basis1 <- basis[,1]
  else
    basis1 <- basis
  obs <- c(obs)
  mask <- which(is.na(obs-basis1))
  if(length(mask)>0){
    recons <- rep(NA, length(obs))
    obs <- obs[-mask]
    if(nb)
      basis <- basis[-mask]
    else
      basis <- basis[-mask,]
    proj <- CalcScores(obs, basis, ...)
    recons.partial <- Recon(proj, basis)
    recons[-mask] <- recons.partial
  }
  else{
    proj <- CalcScores(obs, basis, ...)
    recons <- Recon(proj, basis)
  }
  return(recons)
}





#' Projection onto a basis
#'
#' Calculates the coefficients given by projecting data onto a basis
#'
#' @param data Data matrix to be projected, where each column is a representation on the original field
#' @param basis Basis matrix
#' @param weightinv If NULL, uses standard SVD projection. Otherwise, uses weighted projection.
#'
#' @return Matrix of basis coefficients
#'
#' @examples # First generate some data
#'
#' l <- 100 # dimension of output
#' n <- 10 # number of runs
#' DataBasis <- MakeDataBasis(data = matrix(runif(l*n), nrow=l, ncol=n), RemoveMean = TRUE) # data is 100x10
#'
#' # Project the (centred) ensemble onto the first 3 vectors of the SVD basis
#'
#' Coefficients <- CalcScores(data = DataBasis$CentredField, basis = DataBasis$tBasis[,1:3])
#'
#' # Instead of projecting using W = I, define a W with varying diagonal
#'
#' W <- runif(l, 1, 5) * diag(l) # 100x100 diagonal matrix
#' W_inv <- GetInverse(W) # inverse needed for projection
#' Coefficients_weighted <- CalcScores(data = DataBasis$CentredField, basis = DataBasis$tBasis[,1:3], weightinv = W_inv)
#'
#' @export
Project <- function(data, basis, weightinv = NULL){
  d <- dim(data)[2]
  if (is.null(d)){
    d <- 1
  }
  p <- dim(basis)[2]
  l <- dim(basis)[1]
  if (is.null(p)){
    p <- 1
  }
  if (d == 1){
    data <- as.vector(data)
  }
  if (is.null(weightinv)){
    weightinv <- 0 # just need to set as something that isn't NULL so can give attribute
    attr(weightinv, 'diagonal') <- attr(weightinv, 'identity') <- TRUE
  }
  if (attributes(weightinv)$identity == TRUE){
    V <- t(basis) %*% basis
    Q <- chol(V)
    y <- backsolve(Q, diag(p), transpose = TRUE)
    x <- backsolve(Q, t(basis) %*% data, transpose = TRUE)
    scores <- crossprod(y, x)
  }
  else if (attributes(weightinv)$diagonal == TRUE) {
    V <- t(basis) %*% (diag(weightinv) * basis)
    Q <- chol(V)
    y <- backsolve(Q, diag(p), transpose = TRUE)
    tmp <- t(basis) %*% (diag(weightinv) * data)
    x <- backsolve(Q, tmp, transpose = TRUE)
    scores <- crossprod(y, x)
  }
  else {
    V <- t(basis) %*% weightinv %*% basis
    Q <- chol(V)
    y <- backsolve(Q, diag(p), transpose = TRUE)
    x <- backsolve(Q, t(basis) %*% weightinv %*% data, transpose = TRUE)
    scores <- crossprod(y, x)
  }
  return(t(scores))
}

CalcScores <- Project

#### Rename ####
#' Number of basis vectors required to explain proportion of data
#'
#' Finds the truncated basis that explains a set proportion of the variaiblity in the data.
#'
#' @param basis Basis matrix
#' @param data Data matrix
#' @param vtot The total proportion of variability in the data to be explained by the truncated basis
#' @param weightinv The inverse of W
#'
#' @return The number of basis vectors required to explain vtot of the data.
#'
#' @export
ExplainT <- function(DataBasis, vtot = 0.95, weightinv = NULL){
  v <- 0
  q <- 0
  while (v < vtot & q < dim(DataBasis$tBasis)[2]){
    v <- VarExplained(DataBasis$tBasis[,1:(q+1)], DataBasis$CentredField, weightinv)
    q <- q + 1
  }
  return(q)
}



#' Reconstruction error
#'
#' Calculates the reconstruction error, R_W(basis, obs), of the observations given a basis and W.
#'
#' @param obs The observations
#' @param basis Basis to project and reconstruct the observations with
#' @param weightinv Inverse of weight matrix W. If NULL (default), calculates the mean squared error
#' @param scale If TRUE, scales by the dimension (so analogous to mean squared error)
#'
#' @return The reconstruction error
#'
#' @export
ReconError <- function(obs, basis, weightinv = NULL, scale = TRUE){
  if (is.null(weightinv)){
    weightinv <- 0
    attr(weightinv, 'diagonal') <- attr(weightinv, 'identity') <- TRUE
  }
  field <- ReconObs(obs, basis, weightinv)
  A <- c(obs) - field
  mask <- which(is.na(A))
  if(length(mask)>0){
    A <- A[-mask]
  }
  if (scale == TRUE){
    s <- length(c(obs))-length(mask)
  }
  else {
    s <- 1
  }
  if (attributes(weightinv)$diagonal == FALSE){
    if(length(mask)>0){
      warning("Implicit assumption that weight specified on the full field even though applying a mask to missing obs/ensemble grid boxes")
      weightinv <- weightinv[-mask,-mask]
    }
    wmse <- (t(A) %*% weightinv %*% A)/ s
  }
  else {
    if (attributes(weightinv)$identity == TRUE){
      wmse <- crossprod(A)/ s
    }
    else {
      wmse <- crossprod(A/(1/diag(weightinv)), A)/ s
    }
  }
  return(wmse)
}




#### More flexibility in specification, e.g. v, time allowed, make prior clearer, remove months etc. ####
#' Finding a calibration-optimal basis rotation
#'
#' Given a DataBasis object, observations, matrix W, vector v, applies the optimal rotation algorithm to find a basis more suitable for calibration.
#'
#' @param DataBasis An object containing lxn ensemble data and the lxn basis that will be rotated
#' @param obs A vector of length l with observations on the same scale as the ensemble
#' @param kmax Maximum number of iterations allowed (defaults to 5)
#' @param weightinv Inverse of positive definite weight matrix W. If set = NULL, uses the identity.
#' @param v Vector of minimum proportion of the ensemble data to be explained by the corresponding rotated basis vector
#' @param vtot Minimum proportion of ensemble variability to be explained by the truncated basis
#' @param MaxTime Maximum time (in seconds) to run the optimiser at each iteration.
#'
#' @return \item{tBasis}{Full rotated basis}
#' \item{CentredField}{The ensemble that was passed into the function}
#' \item{EnsembleMean}{The ensemble mean}
#' \item{scaling}{Initial scaling applied to the data}
#' \item{RW}{The reconstruction error after each iteration}
#' \item{VarExp}{The variance explained by each rotated basis vector}
#' \item{Opt}{Linear combination of the basis that gave rotated basis vectors} #### only true for first really ####
#'
#'@examples # First run an ensemble of idealised function fn
#'
#' n <- 60
#' sample <- as.data.frame(2*maximinLHS(n,6) - 1)
#' colnames(sample) <- c("x1","x2","x3","x4","x5","x6")
#' output <- array(c(rep(0,100*n)), dim=c(10,10,n))
#' for (i in 1:n){
#'   output[,,i] <- fn(as.numeric(sample[i,]))
#' }
#' dim(output) <- c(100, n)
#'
#' DataBasis <- MakeDataBasis(data = output, RemoveMean = TRUE)
#'
#' # Define the observations as a known value of x, plus some noise
#'
#' obs <- c(fn(c(0.7,0.01,0.01,0.25,0.8,-0.9)) + rnorm(100, mean = 0, sd = 0.1))
#' obs <- obs - DataBasis$EnsembleMean # centre observations so that comparable to the data
#'
#' # Look at the VarMSEplot for the SVD basis
#'
#' vSVD <- VarMSEplot(DataBasis = DataBasis, obs = obs)
#'
#' # Perform a rotation
#'
#' RotatedBasis <- RotateBasis(DataBasis = DataBasis, obs = obs, kmax = 3)
#'
#' # Editing the variance constraints so that the first vector explains at least 40% of the ensemble, at least 10% for later vectors
#'
#' RotatedBasis <- RotateBasis(DataBasis = DataBasis, obs = obs, kmax = 3, v = c(0.4,0.1,0.1))
#'
#' # So far assumed that W is the identity. Now add structure to W
#'
#'
#'
#' @export
RotateBasis <- function(DataBasis, obs, kmax = 5, weightinv = NULL, v = c(rep(0.1,5)), vtot = 0.95, prior = NULL,
                        StoppingRule = TRUE, MaxTime = 60, ...){
  data <- DataBasis$CentredField
  basis <- DataBasis$tBasis
  if (!dim(data)[1] == dim(basis)[1]){
    stop("Dimension of ensemble and basis (l) are not the same")
  }
  obs <- c(obs)
  if (!length(obs) == dim(basis)[1]){
    stop("Observations not the correct dimension (l)")
  }
  l <- dim(basis)[1]
  n <- dim(data)[2]
  minRw <- ReconError(obs, basis, weightinv)
  if (is.null(prior)){
    prior <- c(1:dim(basis)[2])
  }
  basis <- basis[,prior]
  mse <- var <- numeric(kmax)
  x <- NULL
  new.basis <- NULL
  if (is.null(weightinv)){
    var_sum <- crossprod(c(data))
  }
  else {
    if (attributes(weightinv)$diagonal == TRUE){
      var_sum <- sum(t(data)^2 %*% diag(weightinv))
    }
    else {
      var_sum <- sum(diag(t(data) %*% weightinv %*% data))
    }
  }
  if (is.null(DataBasis$Q)){
    Q <- NULL
    Lambda <- NULL
  }
  else {
    Q <- DataBasis$Q
    Lambda <- DataBasis$Lambda
  }
  for (i in 1:kmax){
    p <- dim(basis)[2]
    if (is.null(weightinv)){
      psi <- t(basis) %*% diag(l) %*% basis
    }
    else {
      psi <- t(basis) %*% weightinv %*% basis
    }
    opt <- GenSA(c(1, rep(0, p-1)), WeightOptim,  lower = rep(-1, p*1),
                 upper = rep(1, p*1), basis = basis, obs = obs, data = data, weightinv = weightinv,
                 v = v[i], newvectors = new.basis, total_sum = var_sum, psi = psi, control = list(max.time = MaxTime), ...)
    best.patterns <- cbind(new.basis, ReconBasis(opt$par, basis))
    rank <- min(n, l)
    basis <- ResidBasis(best.patterns, data, weightinv, Q, Lambda)[,1:rank]
    x <- c(x, opt$par)
    q <- ExplainT(DataBasis, vtot, weightinv)
    mse[i] <- ReconError(obs, basis[,1:q], weightinv)
    var[i] <- VarExplained(basis[,i], data, weightinv)
    new.basis <- cbind(new.basis, basis[,i])
    basis <- basis[,-(1:i)]
    if (round(mse[i],4) == round(minRw,4)) break #### INSERT STOPPING RULE
  }
  new.basis <- cbind(new.basis, basis)[,1:rank]
  return(list(tBasis = new.basis, CentredField = DataBasis$CentredField,
              EnsembleMean = DataBasis$EnsembleMean, scaling = DataBasis$scaling,
              RW = mse, VarExp = var, Opt = x))
}


#' Find the residual basis
#'
#' Given basis vectors and data, calculate the residual basis to complete the basis for the data
#'
#' @param basisvectors Basis vector(s) to project the data onto
#' @param data Data matrix for to find a basis for
#'
#' @return The full basis for the data, i.e. the basis vectors passed into the function, with the residual basis vectors appended
#'
#' @export
ResidBasis <- function(basisvectors, data, weightinv = NULL, ...){
  if (is.null(weightinv)){
    basisvectors <- orthonormalization(basisvectors,basis=FALSE,norm=TRUE)
  }
  else {
    newvector <- basisvectors[,dim(basisvectors)[2]]
    basisvectors[,dim(basisvectors)[2]] <- newvector / as.numeric(sqrt(t(newvector)%*%weightinv %*% newvector))
  }
  l <- dim(data)[1]
  n <- dim(data)[2]
  recons <- matrix(numeric(l*n), nrow=l)
  for (i in 1:n){
    recons[,i] <- ReconObs(data[,i],basisvectors, weightinv)
  }
  resids <- data - recons
  if (is.null(weightinv)){
    svd.resid <- svd(t(resids))
  }
  else {
    svd.resid <- wsvd(t(resids), weightinv = weightinv, ...)
  }
  new.basis <- cbind(basisvectors, svd.resid$v)[,1:min(l,n)]
  return(new.basis)
}

#' Calculating the proportion of data explained by a basis
#'
#' Calculates the proportion of the data that is explained by projection onto a basis.
#'
#' @param basis The basis
#' @param data The data to be explained
#' @param weightinv Inverse of W (identity if NULL)
#' @param total_sum The total sum of squares of the data with respect to W
#' @param psi t(original_basis) %*% weightinv %*% original_basis, where the new basis is a linear combination of some original basis
#' @param basis_lincom Vector of linear combinations (if new basis is a linear combination of some original basis)
#'
#' @return The proportion of variability in the data that is explained by the basis
#'
#' @export
VarExplained <- function(basis, data, weightinv = NULL, total_sum = NULL, psi = NULL, basis_lincom = NULL){
  coeffs <- t(CalcScores(data, basis, weightinv))
  recon <- basis %*% coeffs
  if (is.null(weightinv)){
    explained <- crossprod(c(recon))/crossprod(c(data))
  }
  else {
    if (is.null(psi)){
      if (attributes(weightinv)$diagonal == TRUE){
        explained_num <- sum(t(recon)^2 %*% diag(weightinv))
      }
      else {
        explained_num <- sum(diag(t(recon) %*% weightinv %*% recon))
      }
    }
    else {
      stopifnot(!is.null(basis_lincom))
      explained_num <- t(coeffs) %*% t(basis_lincom) %*% psi %*%
        basis_lincom %*% coeffs
      explained_num <- sum(diag(explained_num))
    }
    #explained_num <- 0
    #for (i in 1:dim(data)[2]){
    #  explained_num <- explained_num + t(recon[,i]) %*% weightinv %*% recon[,i]
    #}
    #explained_den <- 0
    #for (i in 1:dim(data)[2]){
    #  explained_den <- explained_den + t(data[,i]) %*% weightinv %*% data[,i]
    #}
    if (is.null(total_sum)){
      if (attributes(weightinv)$diagonal == TRUE){
        explained_den <- sum(t(data)^2 %*% diag(weightinv))
      }
      else {
        explained_den <- sum(diag(t(data) %*% weightinv %*% data))
      }
    }
    else {
      explained_den <- total_sum
    }
    explained <- explained_num / explained_den
  }
  return(explained)
}



#' Produce a VarMSEplot
#'
#' Calculates and plots the reconstruction error and proportion of variability explained for each truncated basis
#'
#'  @param DataBasis A DataBasis object, containing the data and basis to be used to produce the VarMSEplot
#'  @param obs Observation vector
#'  @param RecVarData if given, the output given by a previous call of VarMSEplot, so that plots can be reproduced without rerunning potentially slow calculations
#'  @param weightinv Inverse of W used for calculating the reconstruction error. If NULL, calculates the mean squared error
#'  @param min.line If TRUE, plots a solid horizontal line at the minimum value of the reconstruction error
#'  @param bound If TRUE, plots a dotted horizontal line at the value of the history matching bound
#'  @param qmax if not NULL, value for the maximum size of the truncated basis to consider. Useful if W is non-diagonal, and either n or l is large
#'
#'  @return A VarMSEplot, and a matrix containing the plotted data, arranged in columns as (Reconstruction error, proportion of variability explained)
#'
#'  @export
VarMSEplot <- function(DataBasis, obs, RecVarData = NULL, weightinv=NULL, min.line=TRUE, bound=TRUE, qmax = NULL, ...){
  if (!is.null(RecVarData)){
    PlotData <- RecVarData
    qmax <- p <- dim(PlotData)[1]
  }
  else {
    p <- dim(DataBasis$tBasis)[2]
    if (is.null(qmax)){
      qmax <- p
    }
    PlotData <- matrix(numeric(qmax*2), nrow=qmax)
    if (!is.null(DataBasis$scaling)){
      PlotData[,1] <- errors(DataBasis$tBasis[,1:qmax], obs, weightinv)*DataBasis$scaling^2
    }
    else {
      PlotData[,1] <- errors(DataBasis$tBasis[,1:qmax], obs, weightinv)
    }
    if (is.null(weightinv)){
      var_sum <- crossprod(c(DataBasis$CentredField))
    }
    else {
      var_sum <- sum(diag(t(DataBasis$CentredField) %*% weightinv %*% DataBasis$CentredField))
    }
    for (i in 1:qmax){
      PlotData[i,2] <- VarExplained(DataBasis$tBasis[,1:i], DataBasis$CentredField, weightinv, total_sum = var_sum)
    }
  }
  if (qmax < p){
    FinalR <- ReconError(obs, DataBasis$tBasis, weightinv)
    PlotData <- rbind(PlotData, c(FinalR, 1))
    plotseq <- c(1:qmax, p)
  }
  else {
    plotseq <- 1:p
  }
  plot(plotseq, PlotData[,1], type="l", col="red",xlab = expression(k), ylab = '', ...)
  mtext(side = 2, line = 2.5, expression(paste("R "[bold(W)], " (", bold(B)[k], ",", bold(z), ")", " / l")), las = 3, cex = 0.8)
  if (min.line)
    abline(h = min(PlotData[,1]), col=alpha("black", 0.7), lty=4)
  if (bound == TRUE){
    abline(h = qchisq(0.995, length(obs))/length(obs), lty=2)
  }
  par(new = TRUE)
  plot(plotseq, PlotData[,2], type="l", axes=FALSE, xlab=NA, ylab=NA, col="blue", ylim=c(0,1))
  axis(side = 4)
  mtext(side = 4, line = 2.5, expression(paste("V(", bold(B)[k], ",", bold(F), ")")), las=3)
  return(PlotData)
}


errors <- function(basis, obs, weightinv=NULL){
  p <- dim(basis)[2]
  err <- numeric(p)
  if (is.null(weightinv)){
    weightinv <- diag(dim(basis)[1])
    attr(weightinv, 'diagonal') <- attr(weightinv, 'identity') <- TRUE
  }
  for (i in 1:p){
    err[i] <- ReconError(obs, basis[,1:i], weightinv)
  }
  return(err)
}



#' Matrix projection
#'
#' Projects a variance matrix onto a given basis
#'
#' @param mat A square matrix to be projected onto the basis
#' @param basis The basis to project with
#' @param weightinv The inverse of positive definite matrix W. If NULL, uses the standard projection, otherwise projects in the norm given by W.
#'
#' @return The projection of the original matrix on the basis.
#'
#' @export
VarProj <- function(mat, basis, weightinv = NULL){
  if (is.null(weightinv)){
    proj <- t(basis) %*% mat %*% basis
  }
  else {
    V <- t(basis) %*% weightinv %*% basis
    Q <- chol(V)
    y <- backsolve(Q, diag(dim(basis)[2]), transpose = TRUE)
    x <- backsolve(Q, t(basis) %*% weightinv, transpose = TRUE)
    comp <- crossprod(y, x)
    proj <- comp %*% mat %*% t(comp)
  }
  return(proj)
}

#' Function used within optimiser for minimisation of the reconstruction error for rotated basis, subject to constraints
#'
#' Given a vector of weights, gives the new basis vector as a linear combination of the original basis, and calculates the reconstruction error, subject to a variability constraint
#'
#' @param x Vector giving the linear combination of the basis to use
#' @param basis The basis that is being rotated
#' @param obs Observation vector
#' @param data Ensemble data
#' @param weightinv Inverse of matrix W
#' @param v The proportion of variability to be explained by the basis vector
#' @param total_sum Common denominator used to calculate VarExplained
#' @param psi As new basis is linear combinarion of original, if pass psi = t(basis) %*% weightinv %*% basis adds efficiency
#' @param newvectors If the reconstruction error should account for any previous basis vectors
#'
#' @return The reconstruction error
#'
#' @export
WeightOptim <- function(x, basis, obs, data, weightinv, v = 0.1, total_sum = NULL, psi = NULL, newvectors = NULL){
  new.basis <- as.vector(tensor(basis, x, 2, 1))
  if (is.null(newvectors) == FALSE){
    new.basis <- cbind(newvectors, new.basis)
  }
  if (is.null(newvectors) == TRUE){
    v_new <- VarExplained(new.basis, data, weightinv, total_sum, psi, basis_lincom = x)
  }
  else {
    v_new <- VarExplained(new.basis[,dim(new.basis)[2]], data, weightinv, total_sum, psi, basis_lincom = x)
  }
  if (v_new < v){
    y <- 999999999
  }
  else {
    y <- ReconError(obs, new.basis, weightinv)
  }
  return(y)
}

#' Idealised function with spatial output
#'
#' A 6 parameter toy function that gives output over a 10x10 field.
#'
#' @param x A vector of 6 values, corresponding to x_1, ..., x_6.
#'
#' @return A 10x10 field.
#'
#' @examples n <- 60
#' sample <- as.data.frame(2*maximinLHS(n,6) - 1)
#' colnames(sample) <- c("x1","x2","x3","x4","x5","x6")
#' output <- array(c(rep(0,100*n)), dim=c(10,10,n))
#' for (i in 1:n){
#'   output[,,i] <- fn(as.numeric(sample[i,]))
#' }
#' dim(output) <- c(100, n) # vectorising spatial output, so each column of the data matrix is a single realisation of the function
#'
#' @export
fn <- function(x){
  basis1 <- basis2 <- basis3 <- basis4 <- basis5 <- basis6 <- basis7 <- basis8 <- matrix(c(rep(0,100)),nrow=10)
  for (i in 1:10){basis1[i,i] <- 1}
  basis1[10,1] <- basis1[10,2] <- basis1[9,1] <- basis1[1,9] <- basis1[1,10] <- basis1[2,10] <- -1
  basis1[8,1] <- basis1[9,2] <- basis1[10,3] <- basis1[1,8] <- basis1[2,9] <- basis1[3,10] <- -1
  for (i in 1:9){basis2[i,i+1] <- 1}
  for (i in 1:8){basis3[1,i] <- 1}
  for (i in 1:5){basis3[2,i] <- 1}
  for (i in 1:3){basis3[3,i] <- 1}
  basis3[9,9] <- basis3[10,10] <- basis3[9,10] <- basis3[10,9] <- basis3[8,9] <- basis3[8,10] <- basis3[7,2] <- basis3[7,3] <- basis3[7,4] <- basis3[8,2] <- basis3[8,3] <- basis3[8,4] <- -1
  basis3[4,1] <- basis3[5,2] <- basis3[5,3] <- basis3[5,4] <- basis3[6,2] <- basis3[6,3] <- basis3[6,4] <- 1
  for (i in 3:7){basis4[10,i] <- basis4[9,i] <- 1}
  basis4[10,3] <- 0
  basis4[6,1] <- basis4[6,2] <- basis4[7,1] <- basis4[6,3] <- basis4[1,1] <- basis4[1,2] <- basis4[2,1] <- -1
  basis4[5,2] <- basis4[5,3] <- basis4[2,3] <- basis4[2,4] <- basis4[3,3] <- 1
  for (i in 8:10){basis5[4,i] <- basis5[5,i] <- basis5[6,i] <- basis5[7,i] <- 1}
  for (i in 6:8){basis5[3,i] <- basis5[2,i] <- basis5[1,i] <- -1}
  basis5[4,7] <- basis5[4,6] <- -1
  basis5[1,8] <- 0
  for (i in 8:9){basis5[i,10] <- -1}
  for (i in 6:8){basis6[i,5] <- basis6[i,6] <- 1}
  basis6[5,6] <- basis6[5,7] <- basis6[8,5] <- basis6[8,4] <- 1
  for (i in 6:7){basis6[i,2] <- basis6[i,3] <- -1}
  for (i in 6:8){basis6[10,i] <- -1}
  basis6[9,9] <- basis6[9,10] <- basis6[1,4] <- basis6[1,5] <- basis6[8,2] <- basis6[6,9] <- -1
  basis7[2,10] <- basis7[3,10] <- basis7[3,9] <- basis7[3,8] <- basis7[10,8] <- basis7[9,8] <- basis7[8,8] <- basis7[9,9] <- basis7[7,6] <- basis7[7,9] <- basis7[5,7] <- basis7[3,2] <- basis7[4,2] <- basis7[1,8] <- basis7[2,8] <- basis7[2,2] <- basis7[8,3] <- basis7[7,3] <- basis7[10,4] <- basis7[9,4] <- basis7[8,7] <- 1
  basis7[7,5] <- basis7[6,5] <- basis7[6,6] <- basis7[7,4] <- basis7[8,1] <- basis7[8,2] <- basis7[1,3] <- basis7[1,4] <- basis7[10,6] <- basis7[10,5] <- basis7[6,9] <- basis7[6,10] <- basis7[1,6] <- basis7[2,6] <- basis7[2,7] <- basis7[10,9] <- -1
  basis8[3,5] <- basis8[3,6] <- basis8[4,5] <- basis8[4,4] <- basis8[5,10] <- basis8[6,10] <- basis8[1,9] <- basis8[9,7] <- 1
  basis8[9,5] <- basis8[8,5] <- basis8[7,5] <- basis8[10,7] <- basis8[7,2] <- basis8[5,4] <- basis8[7,1] <- basis8[10,1] <- basis8[7,10] <- basis8[7,7] <- basis8[5,1] <- basis8[4,3] <- basis8[6,7] <- -1
  fx <- 3*(10*x[2]^2*basis2 + 5*x[3]^2*basis2 + (x[3] + 1.5*x[1]*x[2])*basis3 + 2*x[2]*basis4 + x[3]*x[1]*basis5 +
             (x[2]*x[1])*basis6 + (x[2]^3)*basis7 + ((x[2] + x[3])^2)*basis8 + 2) + 1.5*dnorm(x[4], 0.2, 0.1)*basis1*(x[5]/(1.3+x[6]))
  noise <- matrix(rnorm(100,0,0.05),nrow=10)
  fx <- fx + noise
  return(fx)
}





