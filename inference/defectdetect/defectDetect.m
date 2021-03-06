%MIT License
%Copyright (c) 2019 Sherman Lo

clearvars;
close all;

this = DefectDetectAbsFilterDeg120();
this.run();
this.printResults();
close all;

this = DefectDetectSubRoiAbsFilterDeg120();
this.run();
this.printResults();
close all;

this = DefectDetectSubRoiAbsFilterDeg30();
this.run();
this.printResults();
close all;

this = DefectDetectSubRoiTiFilterDeg30();
this.run();
this.printResults();
close all;

this = DefectDetectSubRoiTiFilterDeg120();
this.run();
this.printResults();
close all;
