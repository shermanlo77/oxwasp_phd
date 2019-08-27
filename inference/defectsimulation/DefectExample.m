%MIT License
%Copyright (c) 2019 Sherman Lo

%CLASS: DEFECT EXAMPLE
%Save an image with a defect without contamination, with contamination and then filtered after
    %contamination. Also saves any ROC properties
%
%Plot the z image before contamination (with positive pixels)
%Plot the z image after contamination
%Plot the z image after filtering (with positive pixels)
%Plot p values after filtering
%Plot the empirical null mean image
%Plot the empirical null std image (with positive pixels)
%Plot the ROC before contamination, after contamination, after filtering
classdef DefectExample < Experiment
  
  properties (SetAccess = private)
    
    nFilter = 4; %number of filters to be investigated
    filterArray; %cell array of filters
    nRoc = 1000; %for roc function
    
    %z statistics (images)
    imageClean; %before contamination
    imageContaminated; %after contamination
    
    isNonNullImage; %boolean image, true if this pixel is a defect (non-null)
    
    %array of false and true negatives (column vector)
    falsePositiveClean;
    truePositiveClean;
    falsePositiveContaminated;
    truePositiveContaminated;
    falsePositiveFilter; %dim 2: for each filter
    truePositiveFilter; %dim 2: for each filter
    
    defectSimulator; %DefectSimulator object
    imageSize; %size of image (2-vector)
    radius; %radius of filter kernel radius
    randStream; %use for seeding the empirical null filter
    
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    function this = DefectExample()
      this@Experiment();
    end
    
    function printResults(this)
      
      directory = fullfile('reports','figures','inference');

      %plot the z images with positive pixels (before contamination)
      fig = LatexFigure.sub();
      imageCleanPlot = Imagesc(this.imageClean);
      zCleanTester = ZTester(this.imageClean);
      zCleanTester.doTest();
      imageCleanPlot.addPositivePixels(zCleanTester.positiveImage);
      imageCleanPlot.plot();
      saveas(fig,fullfile(directory, strcat(this.experimentName,'_imageClean.eps')),'epsc');
      
      %plot z image with contamination, no positive pixels
      fig = LatexFigure.sub();
      imageContaminatedPlot = Imagesc(this.imageContaminated);
      imageContaminatedPlot.setCLim(imageCleanPlot.clim);
      imageContaminatedPlot.plot();
      saveas(fig,fullfile(directory, strcat(this.experimentName,'_imageContaminated.eps')),'epsc');
      
      %for each filter, plot image filtered with positive pixels, p value, empirical null mean, and
          %empricial null std with positive pixels
      for iFilter = 1:this.nFilter
        filter = this.filterArray{iFilter};
        imageFilter = filter.getFilteredImage();
        fig = LatexFigure.sub();
        imageFilterPlot = Imagesc(imageFilter);
        zFilterTester = ZTester(imageFilter);
        zFilterTester.doTest();
        imageFilterPlot.addPositivePixels(zFilterTester.positiveImage);
        imageFilterPlot.setCLim(imageCleanPlot.clim);
        imageFilterPlot.plot();
        saveas(fig,fullfile(directory, strcat(this.experimentName,'_',class(filter), ...
            'imageFiltered.eps')),'epsc');
        
        %p values
        fig = LatexFigure.sub();
        zFilterTester.plotPValues2(~this.isNonNullImage);
        fig.CurrentAxes.XTick = 10.^(0:4);
        saveas(fig,fullfile(directory, strcat(this.experimentName,'_',class(filter), ...
            'pValueFiltered.eps')),'epsc');
        
        %empirical null mean plot
        fig = LatexFigure.sub();
        imageNullMeanPlot = Imagesc(filter.getNullMean());
        imageNullMeanPlot.setCLim(imageCleanPlot.clim);
        imageNullMeanPlot.plot();
        saveas(fig,fullfile(directory, strcat(this.experimentName,'_',class(filter), ...
            'nullMean.eps')),'epsc');
              
        %empirical null std plot with positive pixels
        fig = LatexFigure.sub();
        imageNullStdPlot = Imagesc(filter.getNullStd());
        imageNullStdPlot.setCLim([0,imageCleanPlot.clim(2)]);
        imageNullStdPlot.addPositivePixels(zFilterTester.positiveImage);
        imageNullStdPlot.plot();
        saveas(fig,fullfile(directory, strcat(this.experimentName,'_',class(filter), ...
            'nullStd.eps')),'epsc');
      end
      
      
      %plot ROC
      fig = LatexFigure.sub();
      for iFilter = 1:this.nFilter
        plot(this.falsePositiveFilter(:,iFilter), this.truePositiveFilter(:,iFilter));
        hold on;
      end
      plot(this.falsePositiveClean, this.truePositiveClean, 'k--');
      hold on;
      plot(this.falsePositiveContaminated, this.truePositiveContaminated, 'k--');
      plot([0,1],[0,1],'k:');
      xlabel('false positive rate');
      ylabel('true positive rate');
      legend('empirical','mad mode','meadian iqr','mean var','Location','southeast');
      saveas(fig,fullfile(directory, strcat(this.experimentName,'_roc.eps')),'epsc');

    end
    
  end
  
  methods (Access = protected)
    
    %IMPLEMENTED: SETUP
    function setup(this, defectSimulator, imageSize, radius)
      this.filterArray = cell(this.nFilter, 1);
      this.defectSimulator = defectSimulator;
      this.imageSize = imageSize;
      this.radius = radius;
      this.randStream = this.defectSimulator.randStream;
      this.falsePositiveFilter = zeros(this.nRoc, this.nFilter);
      this.truePositiveFilter = zeros(this.nRoc, this.nFilter);
    end
    
    %IMPLEMENTED: DO EXPERIMENT
    function doExperiment(this)
      
      %get the defected image
      [this.imageContaminated, this.isNonNullImage, this.imageClean] = ...
          this.defectSimulator.getDefectedImage(this.imageSize);
      %work out true and false positive without filtering, before and after contamination
      [this.falsePositiveClean, this.truePositiveClean, ~] = ...
          roc(this.imageClean, this.isNonNullImage, this.nRoc);
      [this.falsePositiveContaminated, this.truePositiveContaminated, ~] = ...
          roc(this.imageContaminated, this.isNonNullImage, this.nRoc);
         
      %for each filter
      for iFilter = 1:this.nFilter
        %get the filter
        switch iFilter
          case 1
            filter = EmpiricalNullFilter(this.radius);
            filter.setSeed(this.randStream.randi([intmin('int32'),intmax('int32')],'int32'));
          case 2
            filter = MadModeNullFilter(this.radius);
            filter.setSeed(this.randStream.randi([intmin('int32'),intmax('int32')],'int32'));
          case 3
            filter = MedianIqrNullFilter(this.radius);
          case 4
            filter = MeanVarNullFilter(this.radius);
        end
        %filter the image and save the filter
        filter.filter(this.imageContaminated);
        this.filterArray{iFilter} = filter;
        %get the roc properties
        [this.falsePositiveFilter(:, iFilter), this.truePositiveFilter(:, iFilter), ~] = ...
            roc(filter.getFilteredImage(), this.isNonNullImage, this.nRoc);
        
        %progress bar
        this.printProgress(iFilter / this.nFilter);
      end

    end
    
  end
  
end

