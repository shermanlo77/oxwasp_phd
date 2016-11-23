library("cplm");

poisson_rate = 500;
gamma_shape = 80;
gamma_rate = 2;

n = 100;

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

estimate = c(poisson_rate = nu, gamma_shape = alpha, gamma_rate = lambda);
print(estimate);