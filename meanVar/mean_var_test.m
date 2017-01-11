n = 1000;

x = normrnd(1000,200,n,1);

shape = 20;
parameter = [-900;-1000000000];

model = MeanVar_GLM_canonical(shape);
y = model.simulate(1./x,parameter);

model.train(y,x,1000);
disp(model.parameter);

x_plot = linspace(500,1500,100);
[variance_prediction, up_error, down_error] = model.predict(x_plot');

figure;
scatter(x,y);
hold on;
plot(x_plot,variance_prediction);
plot(x_plot,up_error);
plot(x_plot,down_error);
