// Stan implementation of Mixture model (Soft KMeans)
data {
  int<lower=0> N; // number of data points
  int<lower=1> D; // number of dimensions
  int<lower=1> K; // number of clusters
  real y[N]; // standardised errors
  vector[D] x[N]; //inputs
}
parameters {
  matrix[K, D] beta; // mixture parameters
 // vector[D] beta[K];
  ordered[K] sigma;// scales of mixture components
}
model {
  sigma ~ lognormal(-1, 1); //priors on sigma
//  for(k in 1:K) beta[k] ~ normal(0, 1); // not sure about priors for beta
  for(k in 1:K) beta[k] ~ normal(0, 5);
  for(n in 1:N) {
    vector[K] lps;
    lps = log_softmax(beta*x[n]);
    for(k in 1:K) {
      lps[k] = lps[k] + normal_lpdf(y[n] | 0, sigma[k]);
    }
    target += log_sum_exp(lps);
  }
}
generated quantities {
  vector[K] mixture_vec[N];
  vector[N] log_lik;
  for(n in 1:N) {
    vector[K] lps;
    lps = log_softmax(beta*x[n]);
    for(k in 1:K) lps[k] = lps[k] + normal_lpdf(y[n]| 0, sigma[k]);
    log_lik[n] = log_sum_exp(lps);
    mixture_vec[n] = softmax(beta*x[n]);
  }  
}
