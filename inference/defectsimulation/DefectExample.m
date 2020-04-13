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
    
    %METHOD: DEFECT EXAMPLE
    function printResults(this)
      
      directory = fullfile('reports','figures','inference');

      %plot the z images with positive pixels (before contamination)
      fig = LatexFigure.subLoose();
      imageCleanPlot = Imagesc(this.imageClean);
      zCleanTester = ZTester(this.imageClean);
      zCleanTester.doTest();
      imageCleanPlot.addPositivePixels(zCleanTester.positiveImage);
      imageCleanPlot.plot();
      imageCleanPlot.removeLabelSpace();
      print(fig,fullfile(directory, strcat(this.experimentName,'_imageClean.eps')),...
          '-depsc','-loose');
      
      %plot z image with contamination, no positive pixels
      fig = LatexFigure.subLoose();
      imageContaminatedPlot = Imagesc(this.imageContaminated);
      imageContaminatedPlot.setCLim(imageCleanPlot.clim);
      imageContaminatedPlot.plot();
      imageContaminatedPlot.removeLabelSpace();
      print(fig,fullfile(directory, strcat(this.experimentName,'_imageContaminated.eps')),...
          '-depsc','-loose');
      %bw plot
      fig = LatexFigure.subLoose();
      imageContaminatedPlot.setToBw();
      imageContaminatedPlot.plot();
      imageContaminatedPlot.removeLabelSpace();
      print(fig,fullfile(directory, strcat(this.experimentName,'_imageContaminatedBW.eps')),...
          '-deps','-loose');
      print(fig,fullfile(directory, strcat(this.experimentName, ...
          '_imageContaminatedBW.tiff')),'-dtiff','-loose');
      
      %for each filter, plot image filtered with positive pixels, p value, empirical null mean, and
          %empricial null std with positive pixels
      for iFilter = 1:this.nFilter
        filter = this.filterArray{iFilter};
        imageFilter = filter.getFilteredImage();
        fig = LatexFigure.subLoose();
        imageFilterPlot = Imagesc(imageFilter);
        zFilterTester = ZTester(imageFilter);
        zFilterTester.doTest();
        imageFilterPlot.addPositivePixels(zFilterTester.positiveImage);
        imageFilterPlot.setCLim(imageCleanPlot.clim);
        imageFilterPlot.plot();
        imageFilterPlot.removeLabelSpace();
        print(fig,fullfile(directory, strcat(this.experimentName,'_',class(filter), ...
            'imageFiltered.eps')),'-depsc','-loose');
        
        %p values
        fig = LatexFigure.sub();
        zFilterTester.plotPValues2(~this.isNonNullImage);
        fig.CurrentAxes.XTick = 10.^(0:4);
        fig.CurrentAxes.YLim(1) = 10.^(-10.5);
        saveas(fig,fullfile(directory, strcat(this.experimentName,'_',class(filter), ...
            'pValueFiltered.eps')),'epsc');
        
        %empirical null mean plot
        fig = LatexFigure.subLoose();
        imageNullMeanPlot = Imagesc(filter.getNullMean());
        imageNullMeanPlot.setCLim(imageCleanPlot.clim);
        imageNullMeanPlot.plot();
        imageNullMeanPlot.removeLabelSpace();
        print(fig,fullfile(directory, strcat(this.experimentName,'_',class(filter), ...
            'nullMean.eps')),'-depsc','-loose');
              
        %empirical null std plot with positive pixels
        fig = LatexFigure.subLoose();
        imageNullStdPlot = Imagesc(filter.getNullStd());
        imageNullStdPlot.setCLim([0,imageCleanPlot.clim(2)]);
        imageNullStdPlot.addPositivePixels(zFilterTester.positiveImage);
        imageNullStdPlot.plot();
        imageNullStdPlot.removeLabelSpace();
        print(fig,fullfile(directory, strcat(this.experimentName,'_',class(filter), ...
            'nullStd.eps')),'-depsc','-loose');
      end
      
      
      %plot ROC
      %thin the rates to make vector figure
      %it should be stairs rather than plot but accurracy is traded for quality of the figures 
      thinningIndex = round(linspace(1, this.imageSize(1)*this.imageSize(2) + 2, 1000));
      
      fig = LatexFigure.sub();
      for iFilter = 1:this.nFilter
        plot(this.falsePositiveFilter(thinningIndex,iFilter), ...
            this.truePositiveFilter(thinningIndex,iFilter));
        hold on;
      end
      plot(this.falsePositiveClean(thinningIndex), this.truePositiveClean(thinningIndex), 'k-.');
      hold on;
      plot(this.falsePositiveContaminated(thinningIndex), ...
          this.truePositiveContaminated(thinningIndex), 'k-.');
      plot([0,1],[0,1],'k:');
      xlabel('false positive rate');
      ylabel('true positive rate');
      legend('empirical null','MADA-mode','meadian IQR','mean var','Location','southeast');
      print(fig,fullfile(directory, strcat(this.experimentName,'_roc.eps')),'-depsc');

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
      nPoints = imageSize(1) * imageSize(2) + 1;
      this.falsePositiveFilter = zeros(nPoints, this.nFilter);
      this.truePositiveFilter = zeros(nPoints, this.nFilter);
    end
    
    %IMPLEMENTED: DO EXPERIMENT
    function doExperiment(this)
      
      %get the defected image
      [this.imageContaminated, this.isNonNullImage, this.imageClean] = ...
          this.defectSimulator.getDefectedImage(this.imageSize);
      %work out true and false positive without filtering, before and after contamination
      [this.falsePositiveClean, this.truePositiveClean, ~] = ...
          roc(this.imageClean, this.isNonNullImage);
      [this.falsePositiveContaminated, this.truePositiveContaminated, ~] = ...
          roc(this.imageContaminated, this.isNonNullImage);
      
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
