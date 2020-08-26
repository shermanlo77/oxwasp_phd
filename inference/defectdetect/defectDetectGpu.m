%NVIDIA GPU REQUIRED FROM HERE
this = DefectDetectSubRoiAbsFilterDeg120Gpu();
this.run();
this.printResults();
close all;

this = DefectDetectSubRoiAbsFilterDeg30Gpu();
this.run();
this.printResults();
close all;

this = DefectDetectSubRoiTiFilterDeg30Gpu();
this.run();
this.printResults();
close all;

this = DefectDetectSubRoiTiFilterDeg120Gpu();
this.run();
this.printResults();
close all;