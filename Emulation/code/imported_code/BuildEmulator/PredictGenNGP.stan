// Stan object to produce predictions from Nonstationary GP emulator in Stan with nugget option
data {
  int<lower=1> N1; // number of ensemble members in design set.
  int<lower=1> N2; // number of ensemble members in validation set.
  int<lower=1> p; // number of inputs.
  int<lower=1> M; // number of posterior draws.
  int<lower=1> Np; // number of regression functions.
  int<lower=1> L; // number of centres of mass.
  
  row_vector[p] X1[N1]; // design matrix.
  row_vector[p] X2[N2]; // validation matrix.
  vector[N1] y1; // vector of simulator evaluations at design matrix.
  matrix[N1, Np] H1; //regression matrix for design matrix.
  matrix[N2, Np] H2; // regression matrix for validation matrix.
  vector[N1] A1[L]; //array of weights for design matrix.
  vector[N2] A2[L]; //array of weights for validation matrix.
  
  vector[Np] beta[M];
  row_vector[p] delta[M, L];
  real sigma[M, L];
  real nugget[M, L];
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
    matrix[N1, N1] K1;
    matrix[N2, N1] K2;
    vector[N1] e;
    vector[N2] Mu;
    vector[N2] Sigma;
    row_vector[p] XS1[N1];
    row_vector[p] XS2[N2];
    
    Mu = H2*beta[m];
    e = y1 - H1*beta[m];
    for(n in 1:N1) XS1[n] = X1[n] ./ delta[m, 1];
    for(n in 1:N2) XS2[n] = X2[n] ./ delta[m, 1];
    K1 = quad_form_diag(cov_exp_quad(XS1, sigma[m, 1], length_scale), A1[1]);
    K2 = diag_pre_multiply(A2[1], diag_post_multiply(cov_exp_quad(XS2, XS1, sigma[m, 1], length_scale), A1[1]));
    Sigma = A2[1] .* A2[1] * pow(sigma[m, 1], 2.0);
    for(l in 2:L) {
      row_vector[p] XSS1[N1];
      row_vector[p] XSS2[N2];
      for(n in 1:N1) XSS1[n] = X1[n] ./ delta[m, l];
      for(n in 1:N2) XSS2[n] = X2[n] ./ delta[m, l];
      K1 = K1 + quad_form_diag(cov_exp_quad(XSS1, sigma[m, l], length_scale), A1[l]);
      K2 = K2 + diag_pre_multiply(A2[l], diag_post_multiply(cov_exp_quad(XSS2, XSS1, sigma[m, l], length_scale), A1[l]));
      Sigma = Sigma + A2[l].* A2[l]*pow(sigma[m, l], 2.0);
    }
    for(i in 1:N1) K1[i, i] = K1[i, i] + nugget[m, sort_indices_desc(A1[, i])[1]];
    for(i in 1:N2) Sigma[i] = Sigma[i] + nugget[m, sort_indices_desc(A2[, i])[1]];
    
    for(n in 1:N2) {
      real mu;
      real sigma_squared;
      row_vector[N1] kt;
      row_vector[N1] k;
      // calculate mean
      kt = K2[n, ];
      k = mdivide_right_spd(kt, K1);
      mu = Mu[n] + k*e;
      sigma_squared = Sigma[n] - k*kt';
      predict_y[m, n] = normal_rng(mu, sqrt(sigma_squared));
    }
  }
  for(k in 1:N2){
    tmeans[k] = mean(predict_y[,k]);
    tsds[k] = sd(predict_y[,k]);
  }
}
