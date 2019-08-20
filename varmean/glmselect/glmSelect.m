%MIT License
%Copyright (c) 2019 Sherman Lo

close all;
clearvars;

this = GlmSelectAicAbsNoFilterDeg30();
this.run();
this.printResults();
close all;

this = GlmSelectAicAbsNoFilterDeg120();
this.run();
this.printResults();
close all;

this = GlmSelectBicAbsNoFilterDeg30();
this.run();
this.printResults();
close all;

this = GlmSelectBicAbsNoFilterDeg120();
this.run();
this.printResults();
close all;


this = GlmSelectAicAbsFilterDeg30();
this.run();
this.printResults();
close all;

this = GlmSelectAicAbsFilterDeg120();
this.run();
this.printResults();
close all;

this = GlmSelectBicAbsFilterDeg30();
this.run();
this.printResults();
close all;

this = GlmSelectBicAbsFilterDeg120();
this.run();
this.printResults();
close all;
