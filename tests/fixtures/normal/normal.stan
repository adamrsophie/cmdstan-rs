// Posterior: mu | y ~ normal(y / 2, sqrt(1 / 2)) = normal(1.5, 0.7071) for y = 3. The prior and
// the likelihood both matter: without the prior the posterior would be normal(3, 1).
data {
  real y;
}

parameters {
  real mu;
}

model {
  mu ~ normal(0, 1);
  y ~ normal(mu, 1);
}

generated quantities {
  // Posterior predictive: normal(1.5, sqrt(1.5)).
  real y_rep = normal_rng(mu, 1);
}
