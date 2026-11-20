data {
  int<lower=0> N;
  int<lower=0> K;
  matrix[N, K] x;
  vector[N] y;
}
parameters {
  real alpha;
  vector[K] beta;
  real<lower=0> sigma;
}
model {
  alpha ~ normal(0, 10);
  beta ~ normal(0, 10);
  sigma ~ exponential(1);
  y ~ normal(alpha + x * beta, sigma);
}
generated quantities {
  vector[N] mu = alpha + x * beta;
}
