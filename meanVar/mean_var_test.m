n = 10000;
n_bin = 10;

x = normrnd(1000,200,n,1);

shape = 20;
parameter = [-900;-1];

model = MeanVar_GLM_canonical(shape);
y = model.simulate(x.^(-1),parameter);

model.train(y,x,1000);
disp(model.parameter);

x_plot = linspace(min(x),max(x),100);
[variance_prediction, up_error, down_error] = model.predict(x_plot');

plotHistogramHeatmap(x,y,nbin);
colormap gray;
hold on;
plot(x_plot,variance_prediction,'r');
plot(x_plot,up_error,'r--');
plot(x_plot,down_error,'r--');
