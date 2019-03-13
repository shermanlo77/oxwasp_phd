subsampleExample(1100:1299, 400:599, 'inferenceSubsample1');
subsampleExample(500:699, 500:699, 'inferenceSubsample2');

function subsampleExample(rowSubsample, colSubsample, name)

  inferenceExample;

  %FIGURE
  %Plot the z image with a rectangle highlighting the subsample
  fig = LatexFigure.sub();
  image_plot = Imagesc(z_image);
  image_plot.plot();
  hold on;
  rectangle('Position', [colSubsample(1), rowSubsample(1), ...
      colSubsample(end)-colSubsample(1)+1, rowSubsample(end)-rowSubsample(1)+1], ...
      'EdgeColor','r','LineStyle','--');
  saveas(fig,fullfile('reports','figures','inference',strcat(name,'_zImage.eps')),'epsc');

  %get the subsample of z statistics and do BH multiple hypothesis testing without any empirical
      %null correction
  zSampleImage = z_image(rowSubsample, colSubsample);
  zSampleVector = reshape(zSampleImage,[],1);
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
  zTester.estimateNull(0, int32(1351727940));
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
  %Plot the frequency density estimate along with the empirical null mean and std
  fig = LatexFigure.sub(); 
  plot(xPlot, numel(zSampleVector)*fHat); %plot density estimate
  hold on;
  %draw the mode as a vertical line
  densityAtMode = parzen.getDensityEstimate(mu0);
  plot([mu0,mu0],[0,numel(zSampleVector)*densityAtMode],'k--');
  %draw the null std as a normpdf with the same curvature as the density estimate at the mode
  %get x values to evalute the null pdf, choose 1 sigma
  xPlotNull = xPlot((xPlot <  mu0+sigma0) & (xPlot >  mu0-sigma0));
  %evalue the null pdf at these points, scale it so that the norm pdf touches the density estimate
  nullPdfPlot = numel(zSampleVector) * densityAtMode * sqrt(2*pi) * sigma0 * ...
      normpdf(xPlotNull, mu0, sigma0) ;
  plot(xPlotNull,  nullPdfPlot, 'r-.');
  %draw a horizontal line of length 2 sigma, y position of that line is where nullPdfPlot start or
    %ends, get the value of the density estimate at mu +/- sigma
  nullAtSigma1 = mean(nullPdfPlot([1,end]));
  %plot line and arrows to represent 2 std
  %triangle are plotted centred at the point when using scatter, move it so that the tip of the
      %triangle touches the point to make it look like an arrow
  offset = 0.1;
  plot([mu0-sigma0,mu0+sigma0],nullAtSigma1*ones(1,2),'k--');
  scatter(mu0-sigma0+offset,nullAtSigma1,'k<','filled');
  scatter(mu0+sigma0-offset,nullAtSigma1,'k>','filled');
  ylabel('frequency density');
  xlabel('z stat');
  saveas(fig,fullfile('reports','figures','inference',strcat(name,'_densityEstimate.eps')),'epsc');

  %SAVE VALUE
  %save the empirical null mean
  fileId = fopen(fullfile('reports','tables',strcat(name,'_nullMean.txt')),'w');
  fprintf(fileId,'%.2f',mu0);
  fclose(fileId);

  %SAVE VALUE
  %save the empirical null std
  fileId = fopen(fullfile('reports','tables',strcat(name,'_nullStd.txt')),'w');
  fprintf(fileId,'%.2f',sigma0);
  fclose(fileId);

  %FIGURE
  %Plot the subsample z statistics along with the positive pixels
  LatexFigure.sub();
  image_plot = Imagesc(zSampleImage);
  image_plot.addPositivePixels(zTester.positiveImage);
  image_plot.plot();

  %SAVE VALUE
  %Save the lower critical boundary for the empirical null BH procedure
  fileId = fopen(fullfile('reports','tables',strcat(name,'_nullCritical1.txt')),'w');
  fprintf(fileId,'%.2f',zCritical(1));
  fclose(fileId);

  %SAVE VALUE
  %Save the upper critical boundary for the empirical null BH procedure
  fileId = fopen(fullfile('reports','tables',strcat(name,'_nullCritical2.txt')),'w');
  fprintf(fileId,'%.2f',zCritical(2));
  fclose(fileId);

  %SAVE VALUE
  %Save the standarised critical boundary for the empirical null BH procedure
  zCritical = norminv(1-zTester.sizeCorrected/2);
  fileId = fopen(fullfile('reports','tables',strcat(name,'_nullNormalisedCritical.txt')),'w');
  fprintf(fileId,'%.2f',zCritical);
  fclose(fileId);

  %FIGURE
  %Plot the histogram of z statistics
  %Also plot the empirical null BH critical boundary
  fig = LatexFigure.sub();
  zTester.plotHistogram2(false);
  saveas(fig,fullfile('reports','figures','inference',strcat(name,'_nullHistogram.txt')),'epsc');

  %FIGURE
  %plot the p values in order
  %also plot the BH critical boundary
  fig = LatexFigure.sub();
  zTester.plotPValues();
  saveas(fig,fullfile('reports','figures','inference',strcat(name,'_nullPValues.txt')),'epsc');
end