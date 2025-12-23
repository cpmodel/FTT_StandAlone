// Stan object to produce posterior samples for parameters.
data {
  int<lower=1> N1;              // number of ensemble members.
  int<lower=1> pact;            // number of active inputs.
  int<lower=1> pinact;          // number of inactive inputs.
  int<lower=1> p;               // number of inputs (total number of inputs).
  int<lower=1> Np;              // number of regression functions.
  int SwitchDelta;              // switch variable for correlation length parameter prior.
  int SwitchNugget;             // switch variable for nugget parameter prior.
  int SwitchSigma;              // switch variable for sigma parameter prior.
  
// parameters for prior specification.
  real SigSq;        //  parameter for sigma prior.
  real SigSqV;       //  parameter for sigma prior.
  real AlphaAct;             //   parameter for delta prior (active).
  real BetaAct;              //   parameter for delta prior (active).
  real AlphaInact;          //    parameter for delta prior (less active).
  real BetaInact;           //    parameter for delta prior (less active).
  real AlphaNugget;         //    parameter for nugget prior.
  real BetaNugget;          //    parameter for nugget prior.
  real AlphaRegress;        //    parameter for regression coeffcient prior.
  real BetaRegress;         //    parameter for regression coeffcient prior.
  real<lower=0> nuggetfix; // Nugget parameter fixed.
//  real UpperLimitNugget;
  
  row_vector[p] X1[N1];   // design matrix.
  vector[N1] y1;          // vector of simulator evaluations at design matrix.
  matrix[N1, Np] H1;      // regression matrix.
}
transformed data{
  real length_scale;
  length_scale = pow(sqrt(2.0), -1 );
}
parameters{
//  real<lower=0, upper=UpperLimitNugget> nugget;
  real<lower=0> nugget;
  real<lower=0> sigma;
  row_vector<lower=0>[p] delta_par;
  vector[Np] beta;
}
transformed parameters{
  row_vector[p] XScaled[N1];
  vector[N1] Mu;
  matrix[N1, N1] Sigma;
  matrix[N1, N1] L;
  for(i in 1:N1) XScaled[i] = X1[i] ./ delta_par;
  //covariance matrix for design set
  Mu = H1*beta;
  Sigma = cov_exp_quad(XScaled, sigma, length_scale);
  for(k in 1:N1) Sigma[k, k] = Sigma[k, k] + nugget;
  L = cholesky_decompose(Sigma);
}
model{
  // Switch variable for prior specification for delta_par
  // '1' same prior specification for delta_par, '2' separate prior specification for
  // 'active' and 'inactive' parameters.
  if(SwitchDelta == 1) delta_par ~ gamma(AlphaAct, BetaAct);
  else {
    for(i in 1:pact) delta_par[i] ~ gamma(AlphaAct, BetaAct);
    for(i in 1:pinact) delta_par[pact+i] ~ gamma(AlphaInact, BetaInact);
    //for(i in 1:pinact) delta_par[i] ~ lognormal(AlphaInact, BetaInact);
    //for(i in 1:pact) delta_par[pinact+i] ~ gamma(AlphaAct, BetaAct);
  }
  // Switch variable for prior specification for nugget
  // '1' concentrated prior around nuggetfix value
  // '2' inverse gamma specification for nugget.
  if(SwitchNugget == 1) nugget ~ normal(nuggetfix, 0.00001);
  else nugget ~ inv_gamma(AlphaNugget, BetaNugget);

  // Switch variable for prior specification for sigma
  // '1' prior specification from Danny's method normal fixed at regression residuals
  // '2' lognormal prior specification
  if(SwitchSigma == 1) sigma ~ normal(SigSq, SigSqV);
  else sigma ~ lognormal(SigSq, SigSqV);

  if(Np > 1) beta[2:Np] ~ normal(AlphaRegress, BetaRegress);
  y1 ~ multi_normal_cholesky(Mu, L);
}
// Define a log likelihood for diagnostics
generated quantities {
  vector[N1] log_lik;
  vector[N1] y_draw;
  y_draw = multi_normal_cholesky_rng(Mu, L);
  for(n in 1:N1)
    log_lik[n] = multi_normal_cholesky_lpdf(y1 | Mu, L);
}
