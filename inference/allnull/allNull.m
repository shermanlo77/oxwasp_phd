%MIT License
%Copyright (c) 2019 Sherman Lo

clearvars;
close all;

this = AllNullGaussianEmpirical();
this.run();
this.printResults();
close all;

this = AllNullGaussianMadMode();
this.run();
this.printResults();
close all;

this = AllNullGaussianMeanVar();
this.run();
this.printResults();
close all;

this = AllNullGaussianMedianIqr();
this.run();
this.printResults();
close all;

this = AllNullPlaneEmpirical();
this.run();
this.printResults();
close all;

this = AllNullPlaneMadMode();
this.run();
this.printResults();
close all;

this = AllNullPlaneMeanVar();
this.run();
this.printResults();
close all;

this = AllNullPlaneMedianIqr();
this.run();
this.printResults();
close all;