nu = 100;
alpha = 10;
lambda = 0.1;
t = 1;

n_sample = 1E5;
Y = poissrnd(nu*t,n_sample,1);
X = gamrnd(Y*alpha,1/lambda);

x = linspace(min(X(X~=0)),max(X),100);
%f = 1/sqrt(2*pi*(alpha+1))*(nu*t*lambda^alpha*alpha)^(1/(2*(alpha+1)))*exp(-nu*t);
k = -nu;
f = exp(k+sum([-(alpha+2)/(2*(alpha+1))*log(x);-x*lambda;(x/alpha).^(alpha/(alpha+1))*(nu*t)^(1/(alpha+1))*(lambda)^(alpha/(alpha+1))*(alpha+1)]));

h = (max(X)-min(X))/100;
area = 0.5*h*(f(1)+f(end)+2*sum(f(2:(end-1))));
f_ = f/area;

figure;
histogram(X,'Normalization','pdf');
hold on;
plot(x,f_);
hold off;

objective = @(parameters)lnL(parameters,t,X);
[mle,lnL_max] = fminunc(objective,[10,10,10]);
%mle = fminsearch(objective,[1,1,1]);
disp(mle);