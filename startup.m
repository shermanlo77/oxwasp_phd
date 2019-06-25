directoryList = {
  fullfile('results')
  fullfile('results','paper')
  fullfile('results','debug')
  fullfile('reports')
  fullfile('reports','figures')
  fullfile('reports','figures','literatureReview')
  fullfile('reports','figures','data')
  fullfile('reports','figures','compoundPoisson')
  fullfile('reports','figures','meanVar')
  fullfile('reports','figures','inference')
};

for i = 1:numel(directoryList)
  directory = directoryList{i};
  if (exist(directory,'dir')==0)
    mkdir(directory);
  end
end

addpath(genpath(fullfile('scan')));
addpath(genpath(fullfile('shadingcorrection')));
addpath(genpath(fullfile('compoundPoisson')));
addpath(genpath(fullfile('regressor')));
addpath(genpath(fullfile('meanVar')));
addpath(genpath(fullfile('inference')));