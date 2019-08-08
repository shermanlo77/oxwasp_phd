%ABSTRACT CLASS: EXPERIMENT DEFECT DETECT
%Compares a test image with aRTist, where the test image is an x-ray scan and aRTist is a simulation
    %of that scan
%In detail: nSample of x-ray images were obtained, all but one are used to train the variance-mean
    %relationship. The remaining image is compared with aRTist, a z image is obtained by subtracting
    %the 2 images and dividing by the sqrt predicted variance given aRTist. The z image is filtered
    %using the empirical null filter. Various kernel radius are investigated in this experiment. The
    %filtered image along with the empirical null mean and std are recorded
classdef (Abstract) DefectDetect < Experiment
  
  properties (SetAccess = protected)
    
    zImage; %the unfiltered z image
    scan; %scan object, containing images of the x-ray scan
    radiusArray; %array of kernel radius to investigate
    testImage; %test projection
    
    zFilterArray; %array of filtered z images, one for each kernel radius
    nullMeanArray; %array of null mean images, one for each kernel radius
    nullStdArray; %array of null std images, one for each kernel radius
    
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    function this = DefectDetect()
      this@Experiment();
    end
    
    %IMPLEMENTED: PRINT RESULTS
    %PARAMETERS:
      %nullStdCLim: cLim for the null std plot, empty to use default min and max null std for cLim
    function printResults(this, zCLim, nullStdCLim, logPMax)
      
      directory = fullfile('reports','figures','inference');
      
      %plot unfilted z image
      fig = LatexFigure.sub();
      imagesc = Imagesc(this.zImage);
      imagesc.plot();
      saveas(fig, fullfile(directory, strcat(this.experimentName,'_unfilteredZ.eps')), 'epsc');
      
      %array of p value images from the filtered z images
      logPArray = zeros(this.scan.height, this.scan.width, numel(this.radiusArray));
      
      %for each radius
      for iRadius = 1:numel(this.radiusArray)
        
        %print radius
        fildId = fopen(fullfile(directory,strcat(this.experimentName,'_radius', ...
            num2str(iRadius),'.txt')),'w');
        fprintf(fildId,'%d',this.radiusArray(iRadius));
        fclose(fildId);
        
        %do the hypothesis test on the filtered image
        filteredImage = this.zFilterArray(:,:,iRadius);
        zTester = ZTester(filteredImage);
        zTester.doTest();
        %save the p value
        logPArray(:,:,iRadius) = -log10(zTester.pImage);
        
        %plot the test image with the significant pixels
        fig = LatexFigure.sub();
        positivePlot = Imagesc(this.testImage);
        positivePlot.addPositivePixels(zTester.positiveImage);
        positivePlot.setDilateSize(2);
        positivePlot.plot();
        ax = fig.Children(1);
        print(fig,fullfile(directory, strcat(this.experimentName,'_radius',num2str(iRadius), ...
            '_sig.eps')),'-depsc','-loose');
        print(fig,fullfile(directory, strcat(this.experimentName,'_radius',num2str(iRadius), ...
            '_sig.tif')),'-dtiff');
        
        %plot the filtered image
        fig = LatexFigure.sub();
        filteredImagePlot = Imagesc(filteredImage);
        filteredImagePlot.setCLim(zCLim);
        filteredImagePlot.plot();
        print(fig,fullfile(directory, strcat(this.experimentName,'_radius',num2str(iRadius), ...
            '_z.eps')),'-depsc','-loose');
        
        %plot the null mean
        fig = LatexFigure.sub();
        nullMeanPlot = Imagesc(this.nullMeanArray(:,:,iRadius));
        nullMeanPlot.setCLim(zCLim);
        nullMeanPlot.plot();
        print(fig,fullfile(directory, strcat(this.experimentName,'_radius',num2str(iRadius), ...
            '_nullMean.eps')),'-depsc','-loose');
        print(fig,fullfile(directory, strcat(this.experimentName,'_radius',num2str(iRadius), ...
            '_nullMean.tif')),'-dtiff');
        
        %plot the null std
        fig = LatexFigure.sub();
        nullStdPlot = Imagesc(this.nullStdArray(:,:,iRadius));
        nullStdPlot.setCLim(nullStdCLim);
        nullStdPlot.plot();
        print(fig,fullfile(directory, strcat(this.experimentName,'_radius',num2str(iRadius), ...
            '_nullStd.eps')),'-depsc','-loose');
        print(fig,fullfile(directory, strcat(this.experimentName,'_radius',num2str(iRadius), ...
            '_nullStd.tif')),'-dtiff');
          
        %plot the -log p values
        fig = LatexFigure.sub();
        pPlot = Imagesc(logPArray(:,:,iRadius));
        pPlot.setCLim([0, logPMax]);
        pPlot.plot();
        print(fig,fullfile(directory, strcat(this.experimentName,'_radius',num2str(iRadius), ...
            '_logp.eps')),'-depsc','-loose');
        print(fig,fullfile(directory, strcat(this.experimentName,'_radius',num2str(iRadius), ...
            '_logp.tif')),'-dtiff');
        
      end
      
    end
    
  end
  
  methods (Access = protected)
    
    %IMPLEMENTED: SETUP
    function setup(this, seed, scan, radiusArray)
      
      %assign member variables
      this.scan = scan;
      this.scan.addShadingCorrectorLinear();
      this.radiusArray = radiusArray;
      this.zFilterArray = nan(scan.height, scan.width, numel(radiusArray));
      this.nullMeanArray = nan(scan.height, scan.width, numel(radiusArray));
      this.nullStdArray = nan(scan.height, scan.width, numel(radiusArray));
      
      %get the z image
      randStream = RandStream('mt19937ar','Seed',seed);
      %get the z statistic
      [this.zImage, this.testImage] = getZImage(scan, randStream);
      
    end
    
    %IMPLEMENTED: DO EXPERIMENT
    %For each radius, filter the zImage with a filter with that radius
    function doExperiment(this)
      %for each radius in radius Array
      for iRadius = 1:numel(this.radiusArray)
        %filter the image
        radius = this.radiusArray(iRadius);
        filter = EmpiricalNullFilter(radius);
        filter.filterRoi(this.zImage, this.scan.getRoiPath);
        %get the resulting filtered image and save it
        this.zFilterArray(:,:,iRadius) = filter.getFilteredImage();
        this.nullMeanArray(:,:,iRadius) = filter.getNullMean();
        this.nullStdArray(:,:,iRadius) = filter.getNullStd();
        %print progress
        this.printProgress(iRadius/numel(this.radiusArray));
      end
    end
    
  end
  
  
end

