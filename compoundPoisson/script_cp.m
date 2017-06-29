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

zero_index = (X==0);
n_0 = sum(zero_index);
X_no_0 = X(~zero_index);

x_range = linspace(min(X_no_0),max(X_no_0),500);
pdf_range = zeros(1,numel(x_range));
pdf_predict_range = zeros(1,numel(x_range));
for i = 1:numel(pdf_range)
    pdf_range(i) = compound_poisson_true.getPdf(x_range(i));
    pdf_predict_range(i) = compound_poisson.getPdf(x_range(i));
end
p_0 = compound_poisson_true.getPdf(0);
p_0_predict = compound_poisson.getPdf(0);

figure;
yyaxis left;
h = histogram(X_no_0,'Normalization','CountDensity');
hold on;
scatter(h.BinWidth/2,n_0/h.BinWidth,50,'filled','b');

plot(x_range,pdf_range*(n),'r-');
scatter(0,n*p_0/h.BinWidth,50,'filled','r');

plot(x_range,pdf_predict_range*(n),'g-');
scatter(h.BinWidth,n*p_0_predict/h.BinWidth,50,'filled','g');

plot([h.BinWidth/2,h.BinWidth/2],[0,n_0/h.BinWidth],'b');
plot([0,0],[0,n*p_0/h.BinWidth],'r');
plot([h.BinWidth,h.BinWidth],[0,n*p_0_predict/h.BinWidth],'g');

xlim([min(X),max(X)]);
y_density_lim = ylim;
ylabel('frequency density');
yyaxis right;
ylim([0,y_density_lim(2)*h.BinWidth]);
ylabel('frequency');
xlabel('support');
legend('Real simulation','Zero simulation','Real density','Zero mass');

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