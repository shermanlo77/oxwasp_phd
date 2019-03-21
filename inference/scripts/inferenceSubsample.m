%SCRIPT: INFERENCE SUB SAMPLE
%A subimage from the z image is taken, tested using the BH procedure corrected for the empirical
    %null
%Plots the following:
  %z image with a rectangle to highlight the subimage
  %histogram with critical boundary, not corrected for empirical null
  %plot density estimate with null mean and null std estimate
  %subsample image with positive pixels
  %histogram with critical boundary, corrected for empirical null
  %p values with critical boundary

clc;
clearvars;
close all;

subsampleExample(1:2000, 1:2000, 'inferenceSubsampleAll'); %whole image
subsampleExample(1100:1299, 400:599, 'inferenceSubsample1'); %subimage with no defect
subsampleExample(500:699, 500:699, 'inferenceSubsample2'); %subimage with defect


%PARAMETERS:
  %rowSubsample: vector of indicies of rows which indiciate the position of the subimage
  %colSubsample: vector of indicies of columns which indiciate the position of the subimage
  %name: prefix used when saving results
function subsampleExample(rowSubsample, colSubsample, name)

  [~, ~, zImage] = inferenceExample();

  %FIGURE
  %Plot the z image with a rectangle highlighting the subsample
  fig = LatexFigure.sub();
  image_plot = Imagesc(zImage);
  image_plot.plot();
  hold on;
  rectangle('Position', [colSubsample(1), rowSubsample(1), ...
      colSubsample(end)-colSubsample(1)+1, rowSubsample(end)-rowSubsample(1)+1], ...
      'EdgeColor','r','LineStyle','--');
  saveas(fig,fullfile('reports','figures','inference',strcat(name,'_zImage.eps')),'epsc');

  %get the subsample of z statistics and do BH multiple hypothesis testing without any empirical
      %null correction
  zSampleImage = zImage(rowSubsample, colSubsample);
  zSampleVector = reshape(zSampleImage,[],1);
  zSampleVector(isnan(zSampleVector)) = [];
  zTester = ZTester(zSampleImage);
  zTester.doTest();
  zCritical = zTester.getZCritical();

  %SAVE VALUE
  %Save the critical boundary for the BH procedure
  fileId = fopen( ...
      fullfile('reports','figures','inference',strcat(name,'_criticalBoundary.txt')),'w');
  fprintf(fileId,'%.2f',zCritical(2));
  fclose(fileId);

  %FIGURE
  %Plot the histogram of the z statistics with the BH critical boundary
  fig = LatexFigure.sub();
  zTester.plotHistogram2(false);
  saveas(fig, fullfile('reports','figures','inference',strcat(name,'_histogram.eps')),'epsc');

  %estimate the empirical null and do the test
  zTester.estimateNull(0, int32(-854868324));
  zTester.doTest();
  
  zCritical = zTester.getZCritical(); %get the critical boundary
  mu0 = zTester.nullMean; %get the empirical null mean
  sigma0 = zTester.nullStd; %get the empirical null std
  
  %define what values of x to plot the density estimate
  xPlot = linspace(min(zSampleVector), max(zSampleVector), 500);
  %get the freqency density estimate
  parzen = Parzen(reshape(zSampleVector,[],1));
  fHat = parzen.getDensityEstimate(xPlot);

  %FIGURE
  %Plot the frequency density estimate along with the empirical null
  fig = LatexFigure.sub(); 
  plot(xPlot, numel(zSampleVector)*fHat); %plot density estimate
  hold on;
  %draw the null std
  nullPdfPlot = numel(zSampleVector) * parzen.getDensityEstimate(mu0) * sqrt(2*pi) * sigma0 * ...
      normpdf(xPlot, mu0, sigma0) ;
  plot(xPlot,  nullPdfPlot, 'r-.');
  legend('density estimate', 'empirical null', 'Location', 'best');
  ylabel('frequency density');
  xlabel('z stat');
  xlim([min(zSampleVector), max(zSampleVector)]);
  saveas(fig,fullfile('reports','figures','inference',strcat(name,'_densityEstimate.eps')),'epsc');

  %SAVE VALUE
  %save the empirical null mean
  fileId = fopen(fullfile('reports','figures','inference',strcat(name,'_nullMean.txt')),'w');
  fprintf(fileId,'%.2f',mu0);
  fclose(fileId);

  %SAVE VALUE
  %save the empirical null std
  fileId = fopen(fullfile('reports','figures','inference',strcat(name,'_nullStd.txt')),'w');
  fprintf(fileId,'%.2f',sigma0);
  fclose(fileId);

  %FIGURE
  %Plot the subsample z statistics along with the positive pixels
  fig = LatexFigure.sub();
  image_plot = Imagesc(zSampleImage);
  image_plot.addPositivePixels(zTester.positiveImage);
  image_plot.plot();
  saveas(fig,fullfile('reports','figures','inference',strcat(name,'_subimagePositive.eps')),'epsc');

  %SAVE VALUE
  %Save the lower critical boundary for the empirical null BH procedure
  fileId = fopen(fullfile('reports','figures','inference',strcat(name,'_nullCritical1.txt')),'w');
  fprintf(fileId,'%.2f',zCritical(1));
  fclose(fileId);

  %SAVE VALUE
  %Save the upper critical boundary for the empirical null BH procedure
  fileId = fopen(fullfile('reports','figures','inference',strcat(name,'_nullCritical2.txt')),'w');
  fprintf(fileId,'%.2f',zCritical(2));
  fclose(fileId);

  %SAVE VALUE
  %Save the standarised critical boundary for the empirical null BH procedure
  zCritical = norminv(1-zTester.sizeCorrected/2);
  fileId = fopen(fullfile('reports','figures','inference', ...
      strcat(name,'_nullNormalisedCritical.txt')),'w');
  fprintf(fileId,'%.2f',zCritical);
  fclose(fileId);
  
  %SAVE VALUE
  %Save the number of positive pixels
  fileId = fopen(fullfile('reports','figures','inference', strcat(name,'_nPositive.txt')),'w');
  fprintf(fileId,'%u',sum(sum(zTester.positiveImage)));
  fclose(fileId);

  %FIGURE
  %Plot the histogram of z statistics
  %Also plot the empirical null BH critical boundary
  fig = LatexFigure.sub();
  zTester.plotHistogram2(false);
  saveas(fig,fullfile('reports','figures','inference',strcat(name,'_nullHistogram.eps')),'epsc');

  %FIGURE
  %plot the p values in order
  %also plot the BH critical boundary
  fig = LatexFigure.sub();
  zTester.plotPValues();
  fig.CurrentAxes.XTick = 10.^(0:5);
  saveas(fig,fullfile('reports','figures','inference',strcat(name,'_nullPValues.eps')),'epsc');
end