clc;
clearvars;
close all;

lambda = 1;
alpha = 10;
beta = 1;

n = 1000;

Y = poissrnd(lambda,n,1); %simulate latent poisson variables
X = gamrnd(Y*alpha,1/beta); %simulate observable gamma

lambda_predict = lambda;
alpha_predict = alpha;
beta_predict = beta;
Y_predict = zeros(n,1);
var_predict = zeros(n,1);

n_step = 10;

lnL_array = zeros(n_step+1,1);
lnL_array(1) = cplnL(X,lambda_predict,alpha_predict,beta_predict);

for step = 1:n_step
    for i = 1:n
        [Y_predict(i),var_predict(i)] = EStep(X(i), lambda_predict, alpha_predict, beta_predict);
    end
    
    [lambda_predict, alpha_predict, beta_predict] = MStep(X, Y_predict, var_predict, alpha_predict, beta_predict);
    
    lnL_array(1+step) = cplnL(X,lambda_predict,alpha_predict,beta_predict);
end
figure;
plot(lnL_array);

% [alpha_grid, beta_grid] = meshgrid(linspace(0.1,2,20),linspace(0.1,2,20));
% lnL_full_grid = alpha_grid;
% for i = 1:numel(lnL_full_grid)
%     for j = 1:n
%         Y_predict(j) = EStep(X(j), lambda, alpha_grid(i),beta_grid(i));
%     end
%     lnL_full_grid(i) = cpFulllnL(X,Y_predict,lambda,alpha_grid(i),beta_grid(i));
% end
% figure;
% surf(alpha_grid, beta_grid, lnL_full_grid);
% xlabel('alpha');
% ylabel('beta');

zero_index = (X==0);
n_0 = sum(zero_index);
X_no_0 = X(~zero_index);

x_range = linspace(min(X_no_0),max(X_no_0),500);
pdf_range = zeros(1,numel(x_range));
pdf_predict_range = zeros(1,numel(x_range));
for i = 1:numel(pdf_range)
    pdf_range(i) = cpPdf(x_range(i),lambda,alpha,beta);
    pdf_predict_range(i) = cpPdf(x_range(i),lambda_predict,alpha_predict,beta_predict);
end
p_0 = exp(-lambda);
p_0_predict = exp(-lambda_predict);

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