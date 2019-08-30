%MIT License
%Copyright (c) 2019 Sherman Lo

%SCRIPT: INFERENCE INTRO
%The z statistics were tested using BH procedure, no empirical null was used
%Plots the following:
  %x ray projection and the aRTist simulation
  %-log10 p values as an image
  %x ray scan with positive pixels
  %histogram with critical boundary
  %p values with critical boundary

clearvars;
close all;

%get the x-ray projection, artist and the z image
randStream = RandStream('mt19937ar', 'Seed', uint32(3538096789));
scan = AbsFilterDeg120();
[zImage, test, artist] = getZImage(scan, randStream);
%do hypothesis testing on the zimage (without empirical null filter)
zTester = ZTester(zImage);
zTester.doTest();

clim = [2.2E4,5.5E4]; %set the clim of the imagesc plots of the x-ray and artist
    
%plot the x ray projection
fig = LatexFigure.sub();
testPlot = Imagesc(test);
testPlot.plot();
testPlot.addScale(scan,'k');
ax = gca;
ax.CLim = clim;
saveas(fig,fullfile('reports','figures','inference',strcat(mfilename,'_scan.eps')),'epsc');

%plot the artist simulation
fig = LatexFigure.sub();
artistPlot = Imagesc(artist);
artistPlot.plot();
artistPlot.addScale(scan,'k');
ax = gca;
ax.CLim = clim;
saveas(fig,fullfile('reports','figures','inference',strcat(mfilename,'_artist.eps')),'epsc');

%plot the p value image
fig = LatexFigure.sub();
pPlot = Imagesc(-log10(zTester.pImage));
pPlot.plot();
pPlot.addScale(scan,'y');
saveas(fig,fullfile('reports','figures','inference',strcat(mfilename,'_logp.eps')),'epsc');

%plot the x-ray projection with critical pixels highlighted
fig = LatexFigure.sub();
sigPlot = Imagesc(test);
sigPlot.addPositivePixels(zTester.positiveImage);
sigPlot.plot();
sigPlot.addScale(scan,'k');
ax = gca;
ax.CLim = clim;
saveas(fig,fullfile('reports','figures','inference', ...
    strcat(mfilename,'_positivePixels.eps')),'epsc');

%plot the histogram with critical boundary
fig = LatexFigure.sub();
zTester.plotHistogram2(true);
saveas(fig,fullfile('reports','figures','inference',strcat(mfilename,'_histogram.eps')),'epsc');

%plot p values with critical boundary
fig = LatexFigure.sub();
zTester.plotPValues();
saveas(fig,fullfile('reports','figures','inference',strcat(mfilename,'_pValue.eps')),'epsc');

%save the critical value
zCritical = zTester.getZCritical();
zCritical = zCritical(2);
fileId = fopen(fullfile('reports','figures','inference',strcat(mfilename,'_critical.txt')),'w');
fprintf(fileId,'%.2f',zCritical);
fclose(fileId);