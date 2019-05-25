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
    trainingIndex; %array of index, points to images use for the variance-mean training
    testIndex; %index, points to a image use for comparing with aRTist
    
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
        positivePlot = Imagesc(this.scan.loadImageStack(this.testIndex));
        positivePlot.addPositivePixels(zTester.positiveImage);
        positivePlot.setDilateSize(2);
        positivePlot.plot();
        saveas(fig,fullfile(directory, strcat(this.experimentName,'_radius',num2str(iRadius), ...
            '_sig.eps')),'epsc');
        
        %plot the filtered image
        fig = LatexFigure.sub();
        filteredImagePlot = Imagesc(filteredImage);
        filteredImagePlot.setCLim(zCLim);
        filteredImagePlot.plot();
        saveas(fig,fullfile(directory, strcat(this.experimentName,'_radius',num2str(iRadius), ...
            '_z.eps')),'epsc');
        
        %plot the null mean
        fig = LatexFigure.sub();
        nullMeanPlot = Imagesc(this.nullMeanArray(:,:,iRadius));
        nullMeanPlot.setCLim(zCLim);
        nullMeanPlot.plot();
        saveas(fig,fullfile(directory, strcat(this.experimentName,'_radius',num2str(iRadius), ...
            '_nullMean.eps')),'epsc');
        
        %plot the null std
        fig = LatexFigure.sub();
        nullStdPlot = Imagesc(this.nullStdArray(:,:,iRadius));
        nullStdPlot.setCLim(nullStdCLim);
        nullStdPlot.plot();
        saveas(fig,fullfile(directory, strcat(this.experimentName,'_radius',num2str(iRadius), ...
            '_nullStd.eps')),'epsc');
          
        %plot the -log p values
        fig = LatexFigure.sub();
        pPlot = Imagesc(logPArray(:,:,iRadius));
        pPlot.setCLim([0, logPMax]);
        pPlot.plot();
        saveas(fig,fullfile(directory, strcat(this.experimentName,'_radius',num2str(iRadius), ...
            '_logp.eps')),'epsc');
        
      end
      
    end
    
  end
  
  methods (Access = protected)
    
    %IMPLEMENTED: SETUP
    function setup(this, seed, scan, radiusArray)
      
      %assign member variables
      this.scan = scan;
      this.scan.addShadingCorrectorBw();
      this.radiusArray = radiusArray;
      this.zFilterArray = nan(scan.height, scan.width, numel(radiusArray));
      this.nullMeanArray = nan(scan.height, scan.width, numel(radiusArray));
      this.nullStdArray = nan(scan.height, scan.width, numel(radiusArray));
      
      %assign random index for the training images (train the var-mean) and the test image
      %the test image is compared with aRTist
      randStream = RandStream('mt19937ar','Seed',seed);
      index = randStream.randperm(scan.nSample);
      nTrain = scan.nSample - 1;
      this.trainingIndex = index(1:nTrain);
      this.testIndex = index(end);
      
      %get the aRTist image
      artist = scan.getArtistImageShadingCorrected('ShadingCorrector',1:scan.whiteIndex);
      
      %get the segmentation image
      segmentation = scan.getSegmentation();
      %get the training images
      trainingStack = scan.loadImageStack(this.trainingIndex);
      %segment the image
      trainingStack = reshape(trainingStack, scan.area, nTrain);
      trainingStack = trainingStack(reshape(segmentation,[],1),:);
      %get the segmented mean and variance greyvalue
      trainingMean = mean(trainingStack,2);
      trainingVar = var(trainingStack,[],2);

      %train glm using the training set mean and variance
      model = GlmGamma(1,IdentityLink());
      model.setShapeParameter((nTrain-1)/2);
      model.train(trainingMean, trainingVar);

      %predict variance given aRTist
      varPredict = reshape(model.predict(reshape(artist,[],1)),scan.height, scan.width);

      %get the test images
      test = scan.loadImageStack(this.testIndex);

      %get the z statistic
      this.zImage = (test - artist) ./ sqrt(varPredict);
      %set non segmented pixels to be nan
      this.zImage(~segmentation) = nan;
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

