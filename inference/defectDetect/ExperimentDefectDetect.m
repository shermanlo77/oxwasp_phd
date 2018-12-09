%ABSTRACT CLASS: EXPERIMENT DEFECT DETECT
%Compares a test image with aRTist, where the test image is an x-ray scan and aRTist is a simulation
    %of that scan
%In detail: nSample of x-ray images were obtained, all but one are used to train the variance-mean
    %relationship. The remaining image is compared with aRTist, a z image is obtained by subtracting
    %the 2 images and dividing by the sqrt predicted variance given aRTist. The z image is filtered
    %using the empirical null filter. Various kernel radius are investigated in this experiment. The
    %filtered image along with the empirical null mean and std are recorded
classdef (Abstract) ExperimentDefectDetect < Experiment
  
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
    function this = ExperimentDefectDetect(experimentName)
      this@Experiment(experimentName);
    end
    
    %IMPLEMENTED: PRINT RESULTS
    function printResults(this)
      
      %for each radius
      for iRadius = 1:numel(this.radiusArray)
        
        %do the hypothesis test on the filtered image
        filteredImage = this.zFilterArray(:,:,iRadius);
        zTester = ZTester(filteredImage);
        zTester.doTest();
        
        %plot the test image with the significant pixels
        figure;
        sigPlot = ImagescSignificant(this.scan.loadImageStack(this.testIndex));
        sigPlot.addSigPixels(zTester.sig_image);
        sigPlot.setDilateSize(2);
        sigPlot.plot();
        
        %plot the filtered image
        figure;
        filteredImagePlot = ImagescSignificant(filteredImage);
        filteredImagePlot.plot();
        
        %plot the null mean
        figure;
        nullMeanPlot = ImagescSignificant(this.nullMeanArray(:,:,iRadius));
        nullMeanPlot.plot();
        
        %plot the null std
        figure;
        nullStdPlot = ImagescSignificant(this.nullStdArray(:,:,iRadius));
        nullStdPlot.setCLim([0,5]);
        nullStdPlot.plot();
        
      end
      
    end
    
  end
  
  methods (Access = protected)
    
    %IMPLEMENTED: SETUP
    function setup(this, seed, scan, radiusArray)
      
      %assign member variables
      this.scan = scan;
      this.scan.addDefaultShadingCorrector();
      this.radiusArray = radiusArray;
      this.zFilterArray = nan(scan.height, scan.width, numel(radiusArray));
      this.nullMeanArray = nan(scan.height, scan.width, numel(radiusArray));
      this.nullStdArray = nan(scan.height, scan.width, numel(radiusArray));
      
      %assign random index for the training images (train the var-mean) and the test image
      %the test image is compared with aRTist
      randStream = RandStream('mt19937ar','Seed',seed);
      index = randStream.randperm(scan.n_sample);
      nTrain = scan.n_sample - 1;
      this.trainingIndex = index(1:nTrain);
      this.testIndex = index(end);
      
      %get the aRTist image
      artist = scan.getShadingCorrectedARTistImage(ShadingCorrector(),1:scan.reference_white);
      
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

