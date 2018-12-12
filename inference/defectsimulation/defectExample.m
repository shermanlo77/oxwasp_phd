%PROCEDURE: DEFECT EXAMPLE
%Produce an image with a defect and contamination, use the empirical null filter to recover the
    %image from the contamination. The hypothesis test (or defect detectiong) are compared before
    %and after contamination.
%Plot the z image before contamination
%Plot the z image after contamination
%Plot the z image after filtering
%Plot the empirical null mean image
%Plot the empirical null std image
%Plot the ROC before contamination, after contamination, after filtering
%PARAMETERS:
  %defectSimulator: simulate a defected and contaminated image
  %imageSize: size of the image
  %radius: radius of the kernel
  %nInitial: number of initial points for Newton-Raphson in empirical null filter
  %directory: where to save the figures
  %prefix: what to prepend the file names with
  %(optional) climNullMean: clim for the plot of the empirical null mean
function defectExample(defectSimulator, imageSize, radius, nInitial, directory, prefix, ...
    climNullMean)

  %get the defected image
  [imageContaminated, isAltImage, imagePreContaminated] = ...
      defectSimulator.getDefectedImage([imageSize, imageSize]);

  filter = EmpiricalNullFilter(radius); %filter it
  filter.setNInitial(nInitial);
  filter.filter(imageContaminated);

  %get the empirical null and the filtered image
  imageFiltered = filter.getFilteredImage();
  nullMean = filter.getNullMean();
  nullStd = filter.getNullStd();

  %get the image pre/post contamination and filtered with significant pixels highlighted
  zTesterPreContaminated = ZTester(imagePreContaminated);
  zTesterPreContaminated.doTest();
  zTesterContaminated = ZTester(imageContaminated);
  zTesterContaminated.doTest();
  zTesterFiltered = ZTester(imageFiltered);
  zTesterFiltered.doTest();
  
  %plot the z images
  fig = LatexFigure.sub();
  imagePlot = ImagescSignificant(imagePreContaminated);
  imagePlot.addSigPixels(zTesterPreContaminated.sig_image);
  climOriginal = imagePlot.clim;
  imagePlot.plot();
  saveas(fig,fullfile(directory, strcat(prefix,'_imagePreContaminated.eps')),'epsc');
  
  fig = LatexFigure.sub();
  imagePlot = ImagescSignificant(imageContaminated);
  imagePlot.addSigPixels(zTesterContaminated.sig_image);
  imagePlot.plot();
  saveas(fig,fullfile(directory, strcat(prefix,'_imageContaminated.eps')),'epsc');
  
  fig = LatexFigure.sub();
  imagePlot = ImagescSignificant(imageFiltered);
  imagePlot.addSigPixels(zTesterFiltered.sig_image);
  imagePlot.setCLim(climOriginal);
  imagePlot.plot();
  saveas(fig,fullfile(directory, strcat(prefix,'_imageFiltered.eps')),'epsc');
  
  %empirical null plot
  fig = LatexFigure.sub();
  imagePlot = ImagescSignificant(nullMean);
  if (nargin == 7)
    imagePlot.setCLim(climNullMean);
  end
  imagePlot.plot();
  saveas(fig,fullfile(directory, strcat(prefix,'_nullMean.eps')),'epsc');
  fig = LatexFigure.sub();
  imagePlot = ImagescSignificant(nullStd);
  imagePlot.setCLim([0,5]);
  imagePlot.plot();
  saveas(fig,fullfile(directory, strcat(prefix,'_nullStd.eps')),'epsc');

  %work out true and false positive, plot ROC
  [falsePositivePreContamination, truePositivePreContamination, ~] = ...
      roc(imagePreContaminated, isAltImage, 100);
  [falsePositiveContamination, truePositiveContamination, ~] = ...
      roc(imageContaminated, isAltImage, 100);
  [falsePositiveFiltered, truePositiveFiltered, ~] = ...
      roc(imageFiltered, isAltImage, 100);
  fig = LatexFigure.sub();
  plot(falsePositivePreContamination, truePositivePreContamination);
  hold on;
  plot(falsePositiveContamination, truePositiveContamination);
  plot(falsePositiveFiltered, truePositiveFiltered);
  plot([0,1],[0,1],'k--');
  xlabel('false positive rate');
  ylabel('true positive rate');
  legend('pre-contamination','contaminated','filtered','Location','southeast');
  saveas(fig,fullfile(directory, strcat(prefix,'_roc.eps')),'epsc');

end

