clc;
clearvars;
close all;

set_up_inference_example;

%put the z image in a tester
z_tester = ZTester(z_image);
%do statistics on the z statistics
z_tester.doTest();
    
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
image_plot = ImagescSignificant(z_image);
image_plot.plot();
saveas(fig,fullfile('reports','figures','inference','initial_artist_z_image.eps'),'epsc');

fig = LatexFigure.sub();
image_plot = ImagescSignificant(-log10(z_tester.p_image));
image_plot.plot();
saveas(fig,fullfile('reports','figures','inference','initial_artist_logp.eps'),'epsc');

%plot the phantom scan with critical pixels highlighted
fig = LatexFigure.main;
image_plot = ImagescSignificant(test);
image_plot.addSigPixels(z_tester.sig_image);
image_plot.plot();
saveas(fig,fullfile('reports','figures','inference','initial_artist_sig_pixels.eps'),'epsc');

%histogram
fig = LatexFigure.sub();
z_tester.plotHistCritical()
saveas(fig,fullfile('reports','figures','inference','initial_artist_z_histo.eps'),'epsc');

%qq plot
fig = LatexFigure.sub();
z_tester.plotQQ();
legend('critical','Location','northwest');
saveas(fig,fullfile('reports','figures','inference','initial_artist_z_qq.eps'),'epsc');

%save the critical value
z_critical = z_tester.getZCritical();
z_critical = z_critical(2);
file_id = fopen(fullfile('reports','tables','initial_artist_z_critical.txt'),'w');
fprintf(file_id,'%.2f',z_critical);
fclose(file_id);