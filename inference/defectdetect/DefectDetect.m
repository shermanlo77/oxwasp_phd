%MIT License
%Copyright (c) 2019 Sherman Lo

%ABSTRACT CLASS: EXPERIMENT DEFECT DETECT
%Compares a test image with aRTist, where the test image is an x-ray projection and aRTist is a 
    %simulation of that projection
%In detail: nSample of x-ray projections were obtained, all but one are used to train the
    %variance-mean relationship. The remaining projection is compared with aRTist, a z image is
    %obtained by subtracting the 2 projections and dividing by the sqrt predicted variance given
    %aRTist. The z image is filtered using the empirical null filter. Various kernel radius are
    %investigated in this experiment. The filtered image along with the empirical null mean and std
    %are recorded
classdef (Abstract) DefectDetect < Experiment
  
  properties (SetAccess = protected)
    
    zImage; %the unfiltered z image
    scan; %scan object, containing images of the x-ray projection
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
      %zCLim: cLim for z plot
      %nullStdCLim: cLim for the null std plot
      %lopPMax: cLim max -log p
      %scaleLength: length of scale bar in cm
    function printResults(this, zCLim, nullStdCLim, logPMax, scaleLength)
      
      directory = fullfile('reports','figures','inference');
      
      %plot unfilted z image
      fig = LatexFigure.subLoose();
      imagesc = Imagesc(this.zImage);
      imagesc.plot();
      imagesc.addScale(this.scan,scaleLength,'y');
      imagesc.removeLabelSpace();
      print(fig, fullfile(directory, strcat(this.experimentName,'_unfilteredZ.eps')), ...
        '-depsc','-loose');
      
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
        fig = LatexFigure.subLoose();
        positivePlot = Imagesc(this.testImage);
        positivePlot.addPositivePixels(zTester.positiveImage);
        positivePlot.setDilateSize(2);
        positivePlot.plot();
        positivePlot.addScale(this.scan,scaleLength,'k');
        positivePlot.removeLabelSpace();
        print(fig,fullfile(directory, strcat(this.experimentName,'_radius',num2str(iRadius), ...
            '_sig.eps')),'-depsc','-loose');
        %plot black white
        fig = LatexFigure.subLoose();
        positivePlot.setToBw();
        positivePlot.plot();
        positivePlot.addScale(this.scan,scaleLength,'k');
        positivePlot.removeLabelSpace();
        print(fig,fullfile(directory, strcat(this.experimentName,'_radius',num2str(iRadius), ...
            '_sigBW.eps')),'-deps','-loose');
        print(fig,fullfile(directory, strcat(this.experimentName,'_radius',num2str(iRadius), ...
            '_sigBW.tiff')),'-dtiff','-loose');
        
        %plot the filtered image
        fig = LatexFigure.subLoose();
        filteredImagePlot = Imagesc(filteredImage);
        filteredImagePlot.setCLim(zCLim);
        filteredImagePlot.plot();
        filteredImagePlot.addScale(this.scan,scaleLength,'y');
        filteredImagePlot.removeLabelSpace();
        print(fig,fullfile(directory, strcat(this.experimentName,'_radius',num2str(iRadius), ...
            '_z.eps')),'-depsc','-loose');
        
        %plot the null mean
        fig = LatexFigure.subLoose();
        nullMeanPlot = Imagesc(this.nullMeanArray(:,:,iRadius));
        nullMeanPlot.setCLim(zCLim);
        nullMeanPlot.plot();
        nullMeanPlot.addScale(this.scan,scaleLength,'y');
        nullMeanPlot.removeLabelSpace();
        print(fig,fullfile(directory, strcat(this.experimentName,'_radius',num2str(iRadius), ...
            '_nullMean.eps')),'-depsc','-loose');
        %set to BW
        fig = LatexFigure.subLoose();
        nullMeanPlot.setToBw();
        nullMeanPlot.plot();
        nullMeanPlot.addScale(this.scan,scaleLength,'y');
        nullMeanPlot.removeLabelSpace();
        print(fig,fullfile(directory, strcat(this.experimentName,'_radius',num2str(iRadius), ...
            '_nullMeanBW.eps')),'-deps','-loose');
        print(fig,fullfile(directory, strcat(this.experimentName,'_radius',num2str(iRadius), ...
            '_nullMeanBW.tiff')),'-dtiff','-loose');
        
        %plot the null std
        fig = LatexFigure.subLoose();
        nullStdPlot = Imagesc(this.nullStdArray(:,:,iRadius));
        nullStdPlot.setCLim(nullStdCLim);
        nullStdPlot.plot();
        nullStdPlot.addScale(this.scan,scaleLength,'y');
        nullStdPlot.removeLabelSpace();
        print(fig,fullfile(directory, strcat(this.experimentName,'_radius',num2str(iRadius), ...
            '_nullStd.eps')),'-depsc','-loose');
        %set to BW
        fig = LatexFigure.subLoose();
        nullStdPlot.setToBw();
        nullStdPlot.plot();
        nullStdPlot.addScale(this.scan,scaleLength,'w');
        nullStdPlot.removeLabelSpace();
        print(fig,fullfile(directory, strcat(this.experimentName,'_radius',num2str(iRadius), ...
            '_nullStdBW.eps')),'-deps','-loose');
        print(fig,fullfile(directory, strcat(this.experimentName,'_radius',num2str(iRadius), ...
            '_nullStdBW.tiff')),'-dtiff','-loose');
          
        %plot the -log p values
        fig = LatexFigure.subLoose();
        pPlot = Imagesc(logPArray(:,:,iRadius));
        pPlot.setCLim([0, logPMax]);
        pPlot.plot();
        pPlot.addScale(this.scan,scaleLength,'y');
        pPlot.removeLabelSpace();
        print(fig,fullfile(directory, strcat(this.experimentName,'_radius',num2str(iRadius), ...
            '_logp.eps')),'-depsc','-loose');
        %set to BW
        fig = LatexFigure.subLoose();
        pPlot.setToBw();
        pPlot.setCLim([0, 3]);
        pPlot.plot();
        pPlot.addScale(this.scan,scaleLength,'y');
        pPlot.removeLabelSpace();
        print(fig,fullfile(directory, strcat(this.experimentName,'_radius',num2str(iRadius), ...
            '_logpBW.eps')),'-deps','-loose');
        print(fig,fullfile(directory, strcat(this.experimentName,'_radius',num2str(iRadius), ...
            '_logpBW.tiff')),'-dtiff','-loose');
        
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
        filter = this.getFilter(radius);
        filter.filterRoi(this.zImage, this.scan.getRoiPath);
        %get the resulting filtered image and save it
        this.zFilterArray(:,:,iRadius) = filter.getFilteredImage();
        this.nullMeanArray(:,:,iRadius) = filter.getNullMean();
        this.nullStdArray(:,:,iRadius) = filter.getNullStd();
        %print progress
        this.printProgress(iRadius/numel(this.radiusArray));
      end
    end
    
    %METHOD: GET FILTER
    %Return the empirical null filter for a given radius, can be overridden
        %to use GPU
    function filter = getFilter(this, radius)
      filter = EmpiricalNullFilter(radius);
    end
    
  end
  
  
end

