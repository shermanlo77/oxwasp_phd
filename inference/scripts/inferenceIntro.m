clc;
clearvars;
close all;

inferenceExample;

zTester = ZTester(z_image);
zTester.doTest();

clim = [2.2E4,5.5E4];
    
%plot the phantom and aRTist image
fig = LatexFigure.sub();
phantom_plot = ImagescSignificant(test);
phantom_plot.plot();
ax = gca;
ax.CLim = clim;
saveas(fig,fullfile('reports','figures','inference',strcat(mfilename,'_scan.eps')),'epsc');

fig = LatexFigure.sub();
phantom_plot = ImagescSignificant(aRTist);
phantom_plot.plot();
ax = gca;
ax.CLim = clim;
saveas(fig,fullfile('reports','figures','inference',strcat(mfilename,'_artist.eps')),'epsc');

fig = LatexFigure.sub();
image_plot = ImagescSignificant(-log10(zTester.pImage));
image_plot.plot();
saveas(fig,fullfile('reports','figures','inference',strcat(mfilename,'_logp.eps')),'epsc');

%plot the phantom scan with critical pixels highlighted
fig = LatexFigure.sub();
image_plot = ImagescSignificant(test);
image_plot.addSigPixels(zTester.positiveImage);
image_plot.plot();
ax = gca;
ax.CLim = clim;
saveas(fig,fullfile('reports','figures','inference', ...
    strcat(mfilename,'_positivePixels.eps')),'epsc');

%histogram
fig = LatexFigure.sub();
zTester.plotHistogram2();
saveas(fig,fullfile('reports','figures','inference',strcat(mfilename,'_histogram.eps')),'epsc');

%qq plot
fig = LatexFigure.sub();
zTester.plotPValues();
saveas(fig,fullfile('reports','figures','inference',strcat(mfilename,'_pValue.eps')),'epsc');

%save the critical value
zCritical = zTester.getZCritical();
zCritical = zCritical(2);
file_id = fopen(fullfile('reports','figures','inference',strcat(mfilename,'_critical.txt')),'w');
fprintf(file_id,'%.2f',zCritical);
fclose(file_id);