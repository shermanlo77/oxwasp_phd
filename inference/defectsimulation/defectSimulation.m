%MIT License
%Copyright (c) 2019 Sherman Lo

close all;
clearvars;

this = DefectAltDustEmpirical();
this.run();
this.printResults();
close all;

this = DefectRadiusLine();
this.run();
this.printResults();
close all;

this = DefectRadiusSquare();
this.run();
this.printResults();
close all;