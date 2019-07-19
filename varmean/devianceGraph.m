%SCRIPT: DEVIANCE GRAPH
%Plot the mean scaled deviance (gamma) vs ratio of value/prediction
%
%y-axis: linear scale
%x-axis: log scale

clearvars;
close all;

xPlot = linspace(-1,1,100);
ratio = 10.^(xPlot);
yPlot = 2*(ratio - 1 - log(ratio));

fig = LatexFigure.sub();
plot(ratio, yPlot);
ax = gca;
ax.XScale = 'log';
xlabel('$y/\widehat{y}$','Interpreter','latex');
ylabel('mean scaled deviance');
saveas(fig, fullfile('reports','figures','varmean', strcat(mfilename,'.eps')), 'epsc');