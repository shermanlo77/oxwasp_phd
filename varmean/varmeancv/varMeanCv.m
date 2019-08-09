clearvars;
close all;

experiments = {...
  VarMeanCvAbsNoFilterDeg30();
  VarMeanCvAbsNoFilterDeg120();
  VarMeanCvAbsFilterDeg30();
  VarMeanCvAbsFilterDeg120();
};

for i = 1:numel(experiments)
  experiments{i}.run();
  experiments{i}.printResults();
end