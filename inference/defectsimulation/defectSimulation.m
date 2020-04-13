%MIT License
%Copyright (c) 2019 Sherman Lo

close all;
clearvars;

this = DefectExampleDust();
this.run();
this.printResults();
close all;

this = DefectExampleLine();
this.run();
this.printResults();
close all;

this = DefectExampleSquare20();
this.run();
this.printResults();
close all;

this = DefectExampleSquare40();
this.run();
this.printResults();
close all;

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
