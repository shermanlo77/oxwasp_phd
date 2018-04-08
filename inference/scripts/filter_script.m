clc;
clearvars;
close all;

set_up_inference_example;

convolution = EmpericalConvolution(z_image,20, 20, [200,200]);
convolution.estimateNull();
convolution.setMask(segmentation);
convolution.doTest();

fig = LatexFigure.sub();
convolution.z_tester.plotHistCritical();
saveas(fig,fullfile('reports','figures','inference','filter_histogram.eps'),'epsc');

fig = LatexFigure.sub();
convolution.z_tester.plotPValues();
saveas(fig,fullfile('reports','figures','inference','filter_p.eps'),'epsc');

fig = LatexFigure.sub();
image_plot = ImagescSignificant(convolution.getZNull());
image_plot.plot();
saveas(fig,fullfile('reports','figures','inference','filter_z_image.eps'),'epsc');

fig = LatexFigure.sub();
image_plot = ImagescSignificant(-log10(convolution.p_image));
image_plot.plot();
saveas(fig,fullfile('reports','figures','inference','filter_p_image.eps'),'epsc');

fig = LatexFigure.main();
image_plot = ImagescSignificant(test);
image_plot.addSigPixels(convolution.sig_image);
image_plot.plot();
saveas(fig,fullfile('reports','figures','inference','filter_sig_image.eps'),'epsc');

fig = LatexFigure.main();
image_plot = ImagescSignificant(convolution.mean_null);
image_plot.plot();
saveas(fig,fullfile('reports','figures','inference','filter_mean_null.eps'),'epsc');