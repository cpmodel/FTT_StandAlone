// Stan object to produce posterior samples for parameters of nonstationary
// GP emulator
data{
  int<lower=1> N1;            // number of ensemble members.
  int<lower=1> pact;          // number of active inputs.
  int<lower=1> pinact;        // number of inactive inputs.
  int<lower=1> p;             // number of inputs. 
  int<lower=1> Np;            // number of regression functions.
  int<lower=1> L;             // number of centres of mass.
  int SwitchDelta;            // switch variable for correlation length parameter prior.
  int SwitchNugget;           // switch variable for nugget parameter prior.
  int SwitchSigma;            // switch variable for sigma parameter prior.
  
  // parameters for prior specification.
  real<lower=0> SigSq;        //  parameter for sigma prior.
  real<lower=0> SigSqV;       //  parameter for sigma prior.
  real AlphaAct;             //   parameter for delta prior (active).
  real BetaAct;              //   parameter for delta prior (active).
  real AlphaInact;          //    parameter for delta prior (less active).
  real BetaInact;           //    parameter for delta prior (less active).
  real AlphaNugget;         //    parameter for nugget prior.
  real BetaNugget;          //    parameter for nugget prior.
  real AlphaRegress;        //    parameter for regression coeffcient prior.
  real BetaRegress;         //    parameter for regression coeffcient prior.
  real<lower=0> nuggetfix; // Nugget parameter fixed.
  real UpperLimitNugget;

  row_vector[p] X1[N1];       // design matrix.
  vector[N1] y1;              // vector of simulator evaluations at design matrix.
  matrix[N1, Np] H1;         //  regression matrix.
  vector[N1] A[L];          // array of weights vectors.
}
transformed data {
  real length_scale;
  length_scale = pow(sqrt(2.0), -1 );
}
parameters{
  real<lower=0> sigma[L];
  //positive_ordered[L] sigma;
  row_vector<lower=0>[p] delta_par[L];
  vector[Np] beta;
  real<lower=0, upper=UpperLimitNugget> nugget[L];
}
transformed parameters{
  vector[N1] Mu; 
  matrix[N1, N1] Sigma;
  matrix[N1, N1] Sigma_cholesky;
  row_vector[p] XScaled1[N1];
  
  for(n in 1:N1) XScaled1[n] = X1[n] ./ delta_par[1];
  Sigma = quad_form_diag(cov_exp_quad(XScaled1, sigma[1], length_scale), A[1]);
  
  for(l in 2:L) {
    row_vector[p] XScaled[N1];
    for(n in 1:N1) XScaled[n] = X1[n] ./ delta_par[l];
    Sigma = Sigma + quad_form_diag(cov_exp_quad(XScaled, sigma[l], length_scale), A[l]);
  }
  for(i in 1:N1) Sigma[i, i] = Sigma[i, i] + nugget[sort_indices_desc(A[, i])[1]];
  Sigma_cholesky = cholesky_decompose(Sigma);
  Mu = H1 * beta;
}
model{
  // Switch variable for prior specification for delta_par
  // '1' same prior specification for delta_par, '2' separate prior specification for
  // 'active' and 'inactive' parameters.
  if(SwitchDelta == 1) for(l in 1:L) delta_par[l] ~ gamma(AlphaAct, BetaAct);
  else {
    for(i in 1:pact) delta_par[, i] ~ gamma(AlphaAct, BetaAct);
    for(i in 1:pinact) delta_par[, pact+i] ~ gamma(AlphaInact, BetaInact);
  }
  // Switch variable for prior specification for nugget
  if(SwitchNugget == 1) nugget ~ normal(nuggetfix, 0.00001);
  else nugget ~ inv_gamma(AlphaNugget, BetaNugget);
  // Switch variable for prior specification for sigma
  // '1' prior specification from Danny's method normal fixed at regression residuals
  // '2' lognormal prior specification
  if(SwitchSigma == 1) sigma ~ normal(SigSq, SigSqV);
  else sigma ~ lognormal(SigSq, SigSqV);
  if(Np > 1) {
    beta[2:Np] ~ normal(AlphaRegress, BetaRegress);
  }
  y1 ~ multi_normal_cholesky(Mu, Sigma_cholesky);
}
generated quantities {
  vector[N1] log_lik;
  for(n in 1:N1)
    log_lik[n] = multi_normal_cholesky_lpdf(y1 | Mu, Sigma_cholesky);
}
