%MIT License
%Copyright (c) 2019 Sherman Lo

clearvars;
close all;

this = VarMeanCvAbsNoFilterDeg30();
this.run();
this.printResults();
close all;

this = VarMeanCvAbsNoFilterDeg120();
this.run();
this.printResults();
close all;

this = VarMeanCvAbsFilterDeg30();
this.run();
this.printResults();
close all;

this = VarMeanCvAbsFilterDeg120();
this.run();
this.printResults();
close all;
