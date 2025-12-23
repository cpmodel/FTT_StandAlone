source('rotation_functions.R')
library(GenSA)
library(lhs)
library(tensor)
library(far)

# Generate data from fn
n <- 60 # ensemble size
sample <- as.data.frame(2*maximinLHS(n,6) - 1)
colnames(sample) <- c("x1","x2","x3","x4","x5","x6")
output <- array(c(rep(0,100*n)), dim=c(10,10,n))
for (i in 1:n){
  output[,,i] <- fn(as.numeric(sample[i,]))
}
dim(output) <- c(100, n) # vectorising spatial output, so each column of the data matrix is a single realisation of the function

# Define observations
obs <- c(fn(c(0.7,0.01,0.01,0.25,0.8,-0.9)))

# Consider 2 different examples: 1) W = identity 2) structured W
DataBasis1 <- MakeDataBasis(data = output, RemoveMean = TRUE)
dim(DataBasis1$tBasis)
sum(DataBasis1$tBasis[,1] * DataBasis1$tBasis[,2]) # orthogonal basis vectors in L2

tmp <- t(runif(100,0.01,5))
W <- t(tmp) %*% tmp + runif(100,0.01,0.1)*diag(100) # some positive definite 100x100 weight matrix

# Need inverse of W, GetInverse does this via cholesky decomposition, and assigns attributes
# (whether diagonal, whether the identity) that speeds up later calculations
Winv <- GetInverse(W)
DataBasis2 <- MakeDataBasis(data = output, weightinv = Winv, RemoveMean = TRUE)
DataBasis2$tBasis[,1] %*% Winv %*% DataBasis2$tBasis[,2] # orthogonal basis vectors in W
DataBasis2$tBasis[,1] %*% Winv %*% DataBasis2$tBasis[,1] # with length 1

# We have stored the eigendecomposition of Winv in DataBasis2
# This was needed for wSVD, and will come in useful when calculating the residual basis at each iteration of the rotation
summary(c(Winv - DataBasis2$Q %*% (diag(DataBasis2$Lambda)) %*% t(DataBasis2$Q)))

# Centre observations by the ensemble mean
obsc <- obs - DataBasis1$EnsembleMean

# Look at the VarMSEplots
v1 <- VarMSEplot(DataBasis = DataBasis1, obs = obsc)
v2 <- VarMSEplot(DataBasis = DataBasis2, obs = obsc, weightinv = Winv)

# Check variances correct
svd_d <- wsvd(t(DataBasis2$CentredField), weightinv = Winv)$d
sum(svd_d[1:2]^2 / sum(svd_d^2)) # matches that given by v2

# Project the (centred) ensemble onto the first q vectors of the SVD basis
q1 <- ExplainT(DataBasis1, vtot = 0.95)
q2 <- ExplainT(DataBasis2, vtot = 0.95, weightinv = Winv)

Coeffs1 <- CalcScores(data = DataBasis1$CentredField, basis = DataBasis1$tBasis[,1:q1])
Coeffs2 <- CalcScores(data = DataBasis2$CentredField, basis = DataBasis2$tBasis[,1:q2], weightinv = Winv)

# Rotate!
# Can set MaxTime low here as have small ensemble and l = 100, although MaxTime=30 or 60 usually fine
RotatedBasis1 <- RotateBasis(DataBasis = DataBasis1, obs = obsc, kmax = 3, v = c(0.4,0.1,0.1), MaxTime = 5)
RotatedBasis2 <- RotateBasis(DataBasis = DataBasis2, obs = obsc, weightinv = Winv, kmax = 3, v = c(0.4,0.1,0.1), MaxTime = 5)
# Resulting vectors are orthogonal in L2, W respectively
sum(RotatedBasis1$tBasis[,1] * RotatedBasis1$tBasis[,4])
RotatedBasis2$tBasis[,1] %*% Winv %*% RotatedBasis2$tBasis[,3]
# Do we minimise R_W/do better than before? Compare SVD vs rotated for each
par(mfrow=c(1,2), mar = c(4,2,2,2))
VarMSEplot(RecVarData = v1, obs = obsc, ylim = c(0,26))
VarMSEplot(DataBasis = RotatedBasis1, obs = obsc, ylim = c(0,26))

VarMSEplot(RecVarData = v2, obs = obsc, weightinv = Winv, ylim = c(0,500))
VarMSEplot(DataBasis = RotatedBasis2, obs = obsc, weightinv = Winv, ylim = c(0,500))

# So now we have a rotated basis that better represents observations (truncation now after we've
# minimised reconstruction error R_W), project onto rotated basis
# Usually require an extra basis vector to explain same proportion
q2rot <- ExplainT(RotatedBasis2, vtot = 0.95, weightinv = Winv)

# Look at some plots of the bases, reconstructions
plot.field <- function(field, dim1, col = rainbow(100,start=0.1,end=0.8), ...){
  require(fields)
  x <- 1:dim1
  y <- 1:dim1
  image.plot(x, y, matrix(field, nrow = dim1), col = col, add = FALSE, ...)
}
# Original basis
par(mfrow=c(2,3),mar=c(4,4,2,2))
zmax <- max(c(abs(DataBasis2$tBasis[,1:q2])))
for (i in 1:q2){
  plot.field(DataBasis2$tBasis[,i], dim1 = 10, zlim = c(-zmax,zmax))
}

# Rotated basis
par(mfrow=c(2,3),mar=c(4,4,2,2))
zmax <- max(c(abs(RotatedBasis2$tBasis[,1:q2rot])))
for (i in 1:q2rot){
  plot.field(RotatedBasis2$tBasis[,i], dim1 = 10, zlim = c(-zmax,zmax))
}

# Reconstruction with original basis
par(mfrow=c(1,3), mar = c(4,4,2,2))
plot.field(obs, dim1 = 10, zlim = c(-10,26), main = "Truth")

ObsRecon <- DataBasis2$EnsembleMean + ReconObs(obsc, DataBasis2$tBasis[,1:q2], weightinv = Winv)
plot.field(ObsRecon, dim1 = 10, zlim = c(-10,26), main = "SVD basis recon")

# Reconstruction with rotated basis
ObsReconRot <- DataBasis2$EnsembleMean + ReconObs(obsc, RotatedBasis2$tBasis[,1:q2rot], weightinv = Winv)
plot.field(ObsReconRot, dim1 = 10, zlim = c(-10,26), main = "Rotated basis recon")

# Then emulate each set of coeffs (each column of RotatedCoeffs)
RotatedCoeffs <- CalcScores(data = RotatedBasis2$CentredField, basis = RotatedBasis2$tBasis[,1:q2rot], weightinv = Winv)
dim(RotatedCoeffs) # number of ensemble members x number of basis vectors to emulate




