lambda = 1;
alpha = 100;
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
p_0 = exp(-lambda);

saddle_point = CompoundPoisson_saddlePoint(1);
saddle_array = saddle_point.getDensity(max(min(X_no_0),0.1),max(X_no_0),500,lambda,alpha,1/gamma);

figure;
yyaxis left;
h = histogram(X_no_0,'Normalization','CountDensity');
hold on;
plot(x_range,pdf_range*(n),'r');
%plot(x_range,n*saddle_array,'g');
plot([h.BinWidth/2,h.BinWidth/2],[0,n_0/h.BinWidth],'b');
plot([0,0],[0,n*p_0/h.BinWidth],'r');
scatter(h.BinWidth/2,n_0/h.BinWidth,50,'filled','b');
scatter(0,n*p_0/h.BinWidth,50,'filled','r');
xlim([min(X),max(X)]);
y_density_lim = ylim;
yyaxis right;
ylim([0,y_density_lim(2)*h.BinWidth]);
