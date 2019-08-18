%MIT License
%Copyright (c) 2019 Sherman Lo

close all;
clearvars;

experiments = {...
  GlmSelectAicAbsNoFilterDeg30();
  GlmSelectAicAbsNoFilterDeg120();
  GlmSelectBicAbsNoFilterDeg30();
  GlmSelectBicAbsNoFilterDeg120();
};

for i = 1:numel(experiments)
  experiments{i}.run();
  experiments{i}.printResults();
end

experiments = {...
  GlmSelectAicAbsFilterDeg30();
  GlmSelectAicAbsFilterDeg120();
  GlmSelectBicAbsFilterDeg30();
  GlmSelectBicAbsFilterDeg120();
};

for i = 1:numel(experiments)
  experiments{i}.run();
  experiments{i}.printResults();
end

