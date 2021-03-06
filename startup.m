%MIT License
%Copyright (c) 2019 Sherman Lo

directoryList = {
  fullfile(getResultsDirectory())
  fullfile('reports')
  fullfile('reports','figures')
  fullfile('reports','figures','data')
  fullfile('reports','figures','compoundpoisson')
  fullfile('reports','figures','varmean')
  fullfile('reports','figures','inference')
  fullfile('debugReport')
};

for i = 1:numel(directoryList)
  directory = directoryList{i};
  if (exist(directory,'dir')==0)
    mkdir(directory);
  end
end

addpath(genpath(fullfile('scan')));
addpath(genpath(fullfile('shadingcorrection')));
addpath(genpath(fullfile('compoundpoisson')));
addpath(genpath(fullfile('varmean')));
addpath(genpath(fullfile('inference')));

clc;
clearvars;