experiments = {...
  VarMeanCvAbsNoFilterNull();
  VarMeanCvAbsNoFilterBw();
  VarMeanCvAbsNoFilterLinear();
  VarMeanCvAbsFilterNull();
  VarMeanCvAbsFilterBw();
  VarMeanCvAbsFilterLinear();
};

for i = 1:numel(experiments)
  experiments{i}.run();
end

experiments = {...
  VarMeanCvAbsNoFilterNull();
  VarMeanCvAbsNoFilterBw();
  VarMeanCvAbsNoFilterLinear();
};

fig = figure;
legendArray = line(0,0);
for i = 1:numel(experiments)
  boxplots = experiments{i}.printResults(-0.1 + ((i-1)*0.1),true);
  legendArray(i) = boxplots.getLegendAx();
end
this = experiments{1};
%retick the x axis
ax = gca;
ax.XTick = 1:numel(this.modelArray);
%label each glm with its name
ax.XTickLabelRotation = 45;
ax.XTickLabel = this.modelArray;
ax.XLim = [0.5,numel(this.modelArray)+0.5];
%label the axis and legend
ylabel('training deviance');
legend(legendArray,{'null','bw','linear'},'Location','bestoutside');

file_name = fullfile('reports','figures','meanVar',strcat(this.experiment_name,'_',stat_name,'.eps'));
file_name(file_name==' ') = '_';
saveas(fig,file_name,'epsc');

fig = figure;
for i = 1:numel(experiments)
  experiments{i}.printResults(-0.1 + ((i-1)*0.1),false);
end