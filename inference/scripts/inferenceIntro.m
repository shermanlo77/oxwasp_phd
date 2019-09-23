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
fig = LatexFigure.subLoose();
testPlot = Imagesc(test);
testPlot.plot();
testPlot.addScale(scan,1,'k');
testPlot.removeLabelSpace();
ax = gca;
ax.CLim = clim;
print(fig,fullfile('reports','figures','inference',strcat(mfilename,'_scan.eps')),...
    '-depsc','-loose');

%plot the artist simulation
fig = LatexFigure.subLoose();
artistPlot = Imagesc(artist);
artistPlot.plot();
artistPlot.addScale(scan,1,'k');
artistPlot.removeLabelSpace();
ax = gca;
ax.CLim = clim;
print(fig,fullfile('reports','figures','inference',strcat(mfilename,'_artist.eps')),...
    '-depsc','-loose');

%plot the p value image
fig = LatexFigure.subLoose();
pPlot = Imagesc(-log10(zTester.pImage));
pPlot.plot();
pPlot.addScale(scan,1,'y');
pPlot.removeLabelSpace();
print(fig,fullfile('reports','figures','inference',strcat(mfilename,'_logp.eps')),...
    '-depsc','-loose');

%plot the x-ray projection with critical pixels highlighted
fig = LatexFigure.subLoose();
sigPlot = Imagesc(test);
sigPlot.addPositivePixels(zTester.positiveImage);
sigPlot.plot();
sigPlot.addScale(scan,1,'k');
sigPlot.removeLabelSpace();
ax = gca;
ax.CLim = clim;
print(fig,fullfile('reports','figures','inference', ...
    strcat(mfilename,'_positivePixels.eps')),'-depsc','-loose');

%plot the histogram with critical boundary
fig = LatexFigure.subLoose();
zTester.plotHistogram2(true);
print(fig,fullfile('reports','figures','inference',strcat(mfilename,'_histogram.eps')),...
    '-depsc','-loose');

%plot p values with critical boundary
fig = LatexFigure.subLoose();
zTester.plotPValues();
print(fig,fullfile('reports','figures','inference',strcat(mfilename,'_pValue.eps')),...
    '-depsc','-loose');

%save the critical value
zCritical = zTester.getZCritical();
zCritical = zCritical(2);
fileId = fopen(fullfile('reports','figures','inference',strcat(mfilename,'_critical.txt')),'w');
fprintf(fileId,'%.2f',zCritical);
fclose(fileId);