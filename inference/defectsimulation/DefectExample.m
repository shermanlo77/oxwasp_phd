%CLASS: DEFECT EXAMPLE
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
  %directory: where to save the figures
  %prefix: what to prepend the file names with
%HOW TO USE:
  %call the method plotExample
  %options available via methods
classdef DefectExample < handle
  
  properties (SetAccess = private)
    
    cLimNullMean; %cLim for null mean plot
    plotDefectNullStd = false; %boolean, plot significant pixels on null std plot if true
    nInitial; %number of initial points for Newton-Raphson in empirical null filter
    nRoc = 1000;
    
  end
  
  methods
    
    %CONSTRUCTOR
    function this = DefectExample()
    end
    
    %METHOD: DEFECT EXAMPLE
    function plotExample(this, defectSimulator, imageSize, radius, directory, prefix)

      %get the defected image
      [imageContaminated, isAltImage, imagePreContaminated] = ...
          defectSimulator.getDefectedImage([imageSize, imageSize]);

      filter = EmpiricalNullFilter(radius); %filter it
      if (~isempty(this.nInitial))
        filter.setNInitial(this.nInitial);
      end
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
      if (~isempty(this.cLimNullMean))
        imagePlot.setCLim(this.cLimNullMean);
      end
      imagePlot.plot();
      saveas(fig,fullfile(directory, strcat(prefix,'_nullMean.eps')),'epsc');
      fig = LatexFigure.sub();
      imagePlot = ImagescSignificant(nullStd);
      imagePlot.setCLim([0,5]);
      if (this.plotDefectNullStd)
        imagePlot.addSigPixels(zTesterFiltered.sig_image);
      end
      imagePlot.plot();
      saveas(fig,fullfile(directory, strcat(prefix,'_nullStd.eps')),'epsc');

      %work out true and false positive, plot ROC
      [falsePositivePreContamination, truePositivePreContamination, ~] = ...
          roc(imagePreContaminated, isAltImage, this.nRoc);
      [falsePositiveContamination, truePositiveContamination, ~] = ...
          roc(imageContaminated, isAltImage, this.nRoc);
      [falsePositiveFiltered, truePositiveFiltered, ~] = ...
          roc(imageFiltered, isAltImage, this.nRoc);
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

    %METHOD: SET N INITIAL
    %number of initial points for Newton-Raphson in empirical null filter
    function setNInitial(this, nInitial)
      this.nInitial = nInitial;
    end
    
    %METHOD: SET CLIM NULL MEAN
    %cLim for null mean plot
    function setCLimNullMean(this, cLimNullMean)
      this.cLimNullMean = cLimNullMean;
    end
    
    %METHOD: SET PLOT DEFECT NULL STD
    %boolean, plot significant pixels on null std plot if true
    function setPlotDefectNullStd(this, plotDefectNullStd)
      this.plotDefectNullStd = plotDefectNullStd;
    end
    
  end
  
end
