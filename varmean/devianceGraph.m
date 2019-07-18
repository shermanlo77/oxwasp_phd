clearvars;
close all;

xPlot = linspace(-1,1,100);
ratio = 10.^(xPlot);
yPlot = 2*(ratio - 1 - log(ratio));

figure;
plot(ratio, yPlot);
ax = gca;
ax.XScale = 'log';
xlabel('$y/\widehat{y}$','Interpreter','latex');
ylabel('scaled deviance');