#### Inputs required ####
# ensemble (n runs, l-dimensional output field), lxn matrix (each run of the model is a column)
# observations (vector of length l)
# weight matrix - lxl variance matrix. We use the observation error + discrepancy variance,
#                 for the parallel to history matching that this gives. Alternatively, can just
#                 ignore, and the code defaults to L_2 projection/SVD (i.e. W = identity)


#### Setting W, W^{-1} ####
# The code defaults to W = identity matrix (i.e. uncorrelated, equal errors)
# We think of W as the sum of the observation error and discrepancy matrices, as this
# gives a parallel between the reconstructon error and history matching, and we can use
# the history match bound to assess whether the basis representation of the observations
# will be ruled out.
# If W is not the identity matrix, calculate the inverse via GetInverse, as this assigns 
# attributes to W^{-1}:
GetInverse(W)
## $diagonal - is W^{-1} a diagonal matrix?
## $identity - is W^{-1} the identity matrix?
# These allow computational improvements to be made in other functions, as when W has structure,
# this is used in projection, calculating the reconstruction error, rotation, etc.


#### Important: if W is NOT a multiple of the identity matrix, then you should project in   ####
##   the norm given by W^{-1}. To do so, each function that has a `weightinv' input currently ##`
##   needs to be passed the output of GetInverse(W). In future, this may be tagged onto the   ##
##   DataBasis object to make things easier.                                                  ##
##   It is possible to project in L_2 (W = identity), even when a structured W is known,      ##
##   however this will not give optimal projections, and in some cases more affect emulation, ##
##   calibration/HM results, etc.                                                             ##


#### Formatting data ####
# MakeDataBasis formats the ensemble (== data) so that later functions can be applied
MakeDataBasis(data, weightinv = NULL, RemoveMean = TRUE)
## $CentredField - the ensemble that the basis will be calculated from. 
##                 If RemoveMean = FALSE, then this is just the ensemble
##                 If RemoveMean = TRUE, the ensemble mean is calculated, and removed
## $EnsembleMean - the ensemble mean, a vector of length l
## $tBasis - the SVD (or weighted SVD basis, if W^{-1} is given), calculated across CentredField
## $Q, $Lambda - from the eigendecomposition of W^{-1}. These are used if calculating the
##               weighted SVD basis, and will be useful later when calculating the residual
##               basis at each iteration of the rotation algorithm


#### Centering observations ####
# If the basis is calculated from the centred ensemble, also need to subtract the ensemble
# mean from the observations
# This is currently a manual operation, e.g.
CentredObs <- TrueObs - DataBasis$EnsembleMean


#### General functions ####
# Given data, basis, observations, possibly W^{-1}, can
## project fields onto a basis, giving set of coefficients
CalcScores(data, basis, weightinv = NULL)
## reconstruct fields from a set of coefficients
Recon(coeffs, basis)
## wrapper that does both operations at once (so can find representation of observations
## on chosen basis)
ReconObs(obs, basis, weightinv = NULL)
## calculate the reconstruction error of the observations (default is to scale by dimension,
## analogous to MSE)
ReconError(obs, basis, weightinv = NULL, scale = TRUE)
## calculating variance explained by a basis/basis vector
VarExplained(basis, data, weightinv = NULL, total_sum = NULL, psi = NULL, basis_lincom = NULL)
## (the total_sum, psi, basis_lincom options are used in the rotation algorithm)


#### Assessing basis quality ####
# VarMSEplots show how reconstruction error changes as add basis vectors, compared to
# the history match bound, so can assess whether, given current ensemble/W, the truncated
# basis would rule out the observations (horizontal dotted line)
VarMSEplot(DataBasis, obs, weightinv=NULL)
# Several other inputs (see other documentation), but fine to be left as defaults


#### Basis rotation ####
# If the truncated SVD basis would rule out the observations, but the remainder of the basis
# does reduce the reconstruction error further (see the VarMSEplot), then rotation can help
RotateBasis(DataBasis, obs, kmax = 5, weightinv = NULL, v = c(rep(0.1,5)), vtot = 0.95, prior = NULL,
                        StoppingRule = TRUE, MaxTime = 60)
# Key inputs:
## DataBasis - object containing $CentredField, $tBasis. This is the basis that the rotation
##             is applied to
## obs - observation vector, needs to be on same scale/same centering as CentredField
## kmax - maximum number of iterations. Generally 1 or 2 is sufficient.
## weightinv - W^{-1}, in form given by GetInverse for efficiency
## v - a vector giving the minimum proportion of ensemble variability to be explained by
##     each new basis vector, to ensure that there's enough signal to allow emulation.
##     Setting this is problem dependent, 10% for each is a good starting point, but there
##     will be cases where this is not possible (high variability across high number of
##     dimensions), so an alternative is to use 0.5*proportion explained by each SVD basis
##     vector. Later versions will include a more automated choice of v.
## vtot - proportion of ensemble variability that the final truncated basis should explain.
##        90%, 95%, 99% common choices.
## prior - generally set as NULL. If n is very large, can speed up optimisation by ignoring
##         some basis vectors, e.g. just rotate first 100 => prior = 1:100
## MaxTime - generally quite fast, 60 seconds is fine (for l < 1000, MaxTime < 30 should be ok)
  
