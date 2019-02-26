this = DefectAltDustEmpirical();
this.run();
this.printResults();
close all;

this = DefectAltDustMadMode();
this.run();
this.printResults();
close all;

this = DefectAltDust0Empirical();
this.run();
%this.printResults();
close all;

this = DefectAltDust0MadMode();
this.run();
%this.printResults();
close all;

this = DefectAltDust0Baseline();
this.run();

this = DefectAltDustBaseline();
this.run();