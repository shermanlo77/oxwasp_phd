clc;
clearvars;
close all;

set_up_inference_example;

zTester = ZTester(z_image);
zTester.doTest();
    
%plot the phantom and aRTist image
fig = LatexFigure.sub();
phantom_plot = ImagescSignificant(test);
phantom_plot.plot();
ax = gca;
ax.CLim = [2.2E4,5.5E4];
saveas(fig,fullfile('reports','figures','inference','initial_artist_scan.eps'),'epsc');

fig = LatexFigure.sub();
phantom_plot = ImagescSignificant(aRTist);
phantom_plot.plot();
ax = gca;
ax.CLim = [2.2E4,5.5E4];
saveas(fig,fullfile('reports','figures','inference','initial_artist_aRTist.eps'),'epsc');

fig = LatexFigure.sub();
image_plot = ImagescSignificant(-log10(zTester.pImage));
image_plot.plot();
saveas(fig,fullfile('reports','figures','inference','initial_artist_logp.eps'),'epsc');

%plot the phantom scan with critical pixels highlighted
fig = LatexFigure.sub();
image_plot = ImagescSignificant(test);
image_plot.addSigPixels(zTester.positiveImage);
image_plot.plot();
saveas(fig,fullfile('reports','figures','inference','initial_artist_sig_pixels.eps'),'epsc');

%histogram
fig = LatexFigure.sub();
zTester.plotHistogram2();
saveas(fig,fullfile('reports','figures','inference','initial_artist_z_histo.eps'),'epsc');

%qq plot
fig = LatexFigure.sub();
zTester.plotPValues();
saveas(fig,fullfile('reports','figures','inference','initial_artist_z_qq.eps'),'epsc');

%save the critical value
zCritical = zTester.getZCritical();
z_critical = z_critical(2);
file_id = fopen(fullfile('reports','tables','initial_artist_z_critical.txt'),'w');
fprintf(file_id,'%.2f',z_critical);
fclose(file_id);