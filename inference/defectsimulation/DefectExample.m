%MIT License
%Copyright (c) 2019 Sherman Lo

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
    
    cLim; %cLim for null mean plot
    plotDefectNullStd = false; %boolean, plot significant pixels on null std plot if true
    nInitial; %number of initial points for Newton-Raphson in empirical null filter
    nRoc = 1000;
    randStream;
    
  end
  
  methods
    
    %CONSTRUCTOR
    function this = DefectExample(randStream)
      this.randStream = randStream;
    end
    
    %METHOD: DEFECT EXAMPLE
    function plotExample(this, defectSimulator, imageSize, radius, directory, prefix)

      %get the defected image
      [imageContaminated, isNonNullImage, imagePreContaminated] = ...
          defectSimulator.getDefectedImage([imageSize, imageSize]);

      filter = EmpiricalNullFilter(radius); %filter it
      filter.setSeed(this.randStream.randi([intmin('int32'),intmax('int32')],'int32'));
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
      fig = LatexFigure.sub();
      zTesterPreContaminated.plotPValues2(~isNonNullImage);
      fig.CurrentAxes.XTick = 10.^(0:4);
      saveas(fig,fullfile(directory, strcat(prefix,'_pValuePreContaminated.eps')),'epsc');
      
      zTesterContaminated = ZTester(imageContaminated);
      zTesterContaminated.doTest();
      
      zTesterFiltered = ZTester(imageFiltered);
      zTesterFiltered.doTest();
      fig = LatexFigure.sub();
      zTesterFiltered.plotPValues2(~isNonNullImage);
      fig.CurrentAxes.XTick = 10.^(0:4);
      saveas(fig,fullfile(directory, strcat(prefix,'_pValueFiltered.eps')),'epsc');

      %plot the z images
      fig = LatexFigure.sub();
      imagePlot = Imagesc(imagePreContaminated);
      imagePlot.addPositivePixels(zTesterPreContaminated.positiveImage);
      climOriginal = imagePlot.clim;
      if (~isempty(this.cLim))
        imagePlot.setCLim(this.cLim);
      end
      imagePlot.plot();
      saveas(fig,fullfile(directory, strcat(prefix,'_imagePreContaminated.eps')),'epsc');

      fig = LatexFigure.sub();
      imagePlot = Imagesc(imageContaminated);
      %imagePlot.addPositivePixels(zTesterContaminated.positiveImage);
      imagePlot.plot();
      saveas(fig,fullfile(directory, strcat(prefix,'_imageContaminated.eps')),'epsc');

      fig = LatexFigure.sub();
      imagePlot = Imagesc(imageFiltered);
      imagePlot.addPositivePixels(zTesterFiltered.positiveImage);
      if (~isempty(this.cLim))
        imagePlot.setCLim(this.cLim);
      else
        imagePlot.setCLim(climOriginal);
      end
      imagePlot.plot();
      saveas(fig,fullfile(directory, strcat(prefix,'_imageFiltered.eps')),'epsc');

      %empirical null mean plot
      fig = LatexFigure.sub();
      imagePlot = Imagesc(nullMean);
      if (~isempty(this.cLim))
        imagePlot.setCLim(this.cLim);
      else
        imagePlot.setCLim(climOriginal);
      end
      imagePlot.plot();
      saveas(fig,fullfile(directory, strcat(prefix,'_nullMean.eps')),'epsc');
      
      %empirical null std plot
      fig = LatexFigure.sub();
      imagePlot = Imagesc(nullStd);
      imagePlot.setCLim([0,climOriginal(2)]);
      if (this.plotDefectNullStd)
        imagePlot.addPositivePixels(zTesterFiltered.positiveImage);
      end
      imagePlot.plot();
      saveas(fig,fullfile(directory, strcat(prefix,'_nullStd.eps')),'epsc');

      %work out true and false positive, plot ROC
      [falsePositivePreContamination, truePositivePreContamination, ~] = ...
          roc(imagePreContaminated, isNonNullImage, this.nRoc);
      [falsePositiveContamination, truePositiveContamination, ~] = ...
          roc(imageContaminated, isNonNullImage, this.nRoc);
      %[falsePositiveFiltered, truePositiveFiltered, ~] = ...
          %roc(imageFiltered, isAltImage, this.nRoc);
      fig = LatexFigure.sub();
      
      for i=1:4
        switch i
          case 1
            filter = EmpiricalNullFilter(radius); %filter it
          case 2
            filter = MadModeNullFilter(radius); %filter it
          case 3
            filter = MedianIqrNullFilter(radius);
          case 4
            filter = MeanVarNullFilter(radius);
        end
        
        if (~isempty(this.nInitial))
          filter.setNInitial(this.nInitial);
        end
        filter.filter(imageContaminated);
        %get the empirical null and the filtered image
        imageFiltered = filter.getFilteredImage();
        [falsePositiveFiltered, truePositiveFiltered, ~] = ...
            roc(imageFiltered, isNonNullImage, this.nRoc);
        plot(falsePositiveFiltered, truePositiveFiltered);
        hold on;
      end
      plot(falsePositivePreContamination, truePositivePreContamination, 'k--');
      hold on;
      plot(falsePositiveContamination, truePositiveContamination, 'k--');
      
      plot([0,1],[0,1],'k:');
      xlabel('false positive rate');
      ylabel('true positive rate');
      legend('empirical','mad mode','meadian iqr','mean var','Location','southeast');
      saveas(fig,fullfile(directory, strcat(prefix,'_roc.eps')),'epsc');

    end

    %METHOD: SET N INITIAL
    %number of initial points for Newton-Raphson in empirical null filter
    function setNInitial(this, nInitial)
      this.nInitial = nInitial;
    end
    
    %METHOD: SET CLIM
    %cLim
    function setCLim(this, cLim)
      this.cLim = cLim;
    end
    
    %METHOD: SET PLOT DEFECT NULL STD
    %boolean, plot significant pixels on null std plot if true
    function setPlotDefectNullStd(this, plotDefectNullStd)
      this.plotDefectNullStd = plotDefectNullStd;
    end
    
  end
  
end

