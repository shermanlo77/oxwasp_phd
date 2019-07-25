library("cplm");

n = 100;
poisson_rate = 1;
gamma_shape = 1;
gamma_rate = 1;

n_repeat = 100;

nu_array = rep(0,n_repeat);
alpha_array = rep(0,n_repeat);
lambda_array = rep(0,n_repeat);

for (i in seq_len(n_repeat)){

  Y = rpois(n, poisson_rate);
  
  X = data.frame(x=rgamma(n, gamma_shape*Y, gamma_rate));
  
  fit = cpglm(x~1, link = "identity", X);
  
  phi = fit$phi
  p = fit$p
  mu = fit$coefficients
  names(mu) = NULL;
  
  alpha = (2-p)/(p-1);
  nu = mu^(2-p)/(phi*(2-p));
  lambda = 1/(phi*(p-1)*mu^(p-1));
  
  nu_array[i] = nu;
  alpha_array[i] = alpha;
  lambda_array[i] = lambda;
}

hist(nu_array);
hist(alpha_array);
hist(lambda_array);

