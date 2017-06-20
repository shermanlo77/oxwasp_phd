lambda = 1;
alpha = 1;
gamma = 1;

n = 10000;

Y = poissrnd(lambda,n,1); %simulate latent poisson variables
X = gamrnd(Y*alpha,gamma); %simulate observable gamma

zero_index = (X==0);
n_0 = sum(zero_index);
X_no_0 = X(~zero_index);

x_range = linspace(min(X_no_0),max(X_no_0),500);
pdf_range = zeros(1,numel(x_range));
for i = 1:numel(pdf_range)
    pdf_range(i) = cpPdf(x_range(i),lambda,alpha,gamma);
end

figure;
yyaxis left;
h = histogram(X_no_0,'Normalization','CountDensity');
hold on;
plot(x_range,pdf_range*(n-n_0),'r');
plot([0,0],[0,n_0/h.BinWidth],'b','LineWidth',3);
scatter(0,n_0/h.BinWidth,50,'filled','b');
xlim([min(X),max(X)]);
y_density_lim = ylim;
yyaxis right;
ylim([0,y_density_lim(2)*h.BinWidth]);
