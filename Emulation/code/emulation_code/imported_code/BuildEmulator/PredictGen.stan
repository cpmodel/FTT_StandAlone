// Stan object to produce predictions in Stan with nugget option
data {
  int<lower=1> N1; // number of ensemble members in design set.
  int<lower=1> N2; // number of ensemble members in validation set.
  int<lower=1> p; // number of inputs.
  int<lower=1> M; // number of posterior draws.
  int<lower=1> Np; // number of regression functions.
  
  row_vector[p] X1[N1]; // design matrix.
  row_vector[p] X2[N2]; // validation matrix.
  vector[N1] y1; // vector of simulator evaluations at design matrix.
  matrix[N1, Np] H1; // regression matrix for design matrix.
  matrix[N2, Np] H2; // regression matrix for validation matrix.
  
  real<lower=0> sigma[M];
  real<lower=0> nugget[M];
  row_vector[Np] beta[M];
  row_vector[p] delta[M];
}
transformed data {
  real length_scale;
  length_scale = pow(sqrt(2.0), -1);
}
model {
}
generated quantities {
  matrix[M, N2] predict_y;
  vector[N2] tmeans;
  vector[N2] tsds;
  for(m in 1:M) {
    row_vector[p] XS1[N1];
    row_vector[p] XS2[N2];
    matrix[N1, N1] A1;
    matrix[N2, N1] A2;
    matrix[N1, N1] inver;
    vector[N1] e;
    
    // scaled design matrix by correlation length parameters.
    for(i in 1:N1) XS1[i] = X1[i] ./ delta[m];
    // scaled validation matrix by correlation length parameters.
    for(i in 1:N2) XS2[i] = X2[i] ./ delta[m];
    // covariance matrix for design set. 
    A1 = cov_exp_quad(XS1, sigma[m], length_scale);
    for(i in 1:N1) A1[i, i] = A1[i, i] + nugget[m];
    // inverse of covariance matrix for design set.
    inver = inverse(A1);
    // covariance matrix for validation set.
    A2 = cov_exp_quad(XS2, XS1, sigma[m], length_scale);
    // error vector.
    e = y1 - H1*to_vector(beta[m]);
    for(n in 1:N2) {
      real mu;
      real sigma_squared;
      // calculate mean
      mu = H2[n, ]*to_vector(beta[m]) + A2[n, ]*inver*e;
      // calculate variance
      sigma_squared = pow(sigma[m], 2.0) + nugget[m] - A2[n, ]*inver*A2[n, ]';
      predict_y[m, n] = normal_rng(mu, sqrt(sigma_squared));
    }
  }
  for(k in 1:N2) {
    tmeans[k] = mean(predict_y[, k]);
    tsds[k] = sd(predict_y[, k]);
  }
}








