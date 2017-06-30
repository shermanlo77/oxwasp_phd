clc;
clearvars;
close all;

lambda = 1;
alpha = 1;
beta = 1;

n = 100;

[X,Y] = CompoundPoisson.simulate(n,lambda,alpha,beta);

compound_poisson = CompoundPoisson();
compound_poisson.addData(X);
compound_poisson.initaliseEM();
compound_poisson.setParameters(lambda,alpha,beta);

compound_poisson_true = CompoundPoisson();
compound_poisson_true.setParameters(lambda,alpha,beta);

n_step = 10;

lnL_array = zeros(n_step+1,1);
lnL_array(1) = compound_poisson.getMarginallnL();

for step = 1:n_step
    compound_poisson.EStep();
    compound_poisson.MStep();    
    lnL_array(1+step) = compound_poisson.getMarginallnL();
end
figure;
plot(lnL_array);


[alpha_grid, beta_grid] = meshgrid(linspace(0.1,2,20),linspace(0.1,2,20));
lnL_full_grid = alpha_grid;
for i = 1:numel(lnL_full_grid)
    compound_poisson.setParameters(lambda,alpha_grid(i),beta_grid(i));
    compound_poisson.EStep();
    lnL_full_grid(i) = compound_poisson.getMObjective();
end
figure;
surf(alpha_grid, beta_grid, lnL_full_grid);
xlabel('alpha');
ylabel('beta');