%CLASS: EXPERIMENT ALL GAUSSIAN
%See how the empirical null filter behaves for images of pure Gaussian noise
%A Gaussian noise image is produced and then filtered using the empirical null filter. Various
%  properties of the post-filtered image are recorded such as the mean, variance, p-value from the
%  KS test and the time it took to filter the image.
%This is repeated nRepeat times for various kernel radius
%Plots the following:
%  post-filter mean vs radius
%  post-filter variance vs radius
%  log ks p value vs radius
%  time to filter vs radius
classdef Experiment_AllGaussian < Experiment
  
  properties (SetAccess = private)
    
    %array of kernel radius to investigate
    radiusArray = 10:10:100;
    %number of times to repeat the experiment
    nRepeat = 100;
    %number of initial points
    nInitial = 3;
    
    %array to store results of the post filtered image
      %dim 1: for each n repeat
      %dim 2: for each radius
    meanArray; %mean of all pixels
    varianceArray; %variance of all pixels
    ksArray; %kolmogorov-smirnov p value
    timeArray; %time to filter the image in seconds
    
    imageSize = [256, 256]; %size of the gaussian image
    
    randStream = RandStream('mt19937ar','Seed',uint32(3499211588)); %rng
    
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    function this = Experiment_AllGaussian()
      this@Experiment("Experiment_AllGaussian");
    end
    
    %METHOD: PRINT RESULTS
    function printResults(this)
      
      sigma = 2; %get sigma level
      alpha = 2*(1-normcdf(sigma)); %covert to significant level
      n = this.imageSize(1) * this.imageSize(2); %get number of pixels in the image
      
      %plot post filter mean vs radius
      figure;
      meanPlot = Boxplots(this.meanArray, true);
      meanPlot.setPosition(this.radiusArray);
      meanPlot.plot();
      hold on;
      meanCritical = sigma/sqrt(n);
      plot([0,this.radiusArray(end)+10],[meanCritical,meanCritical], 'k--');
      plot([0,this.radiusArray(end)+10],[-meanCritical,-meanCritical], 'k--');
      xlim([0,this.radiusArray(end)+10]);
      ylabel('post filter image greyvalue mean');
      xlabel('radius (pixel)');
      
      %plot post filter variance vs radius
      figure;
      varPlot = Boxplots(this.varianceArray, true);
      varPlot.setPosition(this.radiusArray);
      varPlot.plot();
      hold on;
      varCritical1 = chi2inv(alpha, n-1)/(n-1);
      varCritical2 = chi2inv(1-alpha, n-1)/(n-1);
      plot([0,this.radiusArray(end)+10],[varCritical1,varCritical1], 'k--');
      plot([0,this.radiusArray(end)+10],[varCritical2,varCritical2], 'k--');
      xlim([0,this.radiusArray(end)+10]);
      ylabel('post filter image greyvalue variance');
      xlabel('radius (pixel)');
      
      %plot log ks p value vs radius
      figure;
      ksArrayCopy = this.ksArray();
      ksArrayCopy(ksArrayCopy<0) = nan;
      ksPlot = Boxplots(log10(ksArrayCopy), true);
      ksPlot.setPosition(this.radiusArray);
      ksPlot.plot();
      hold on;
      plot([0,this.radiusArray(end)+10],log10([alpha,alpha]), 'k--');
      xlim([0,this.radiusArray(end)+10]);
      ylabel('KS log p-value');
      xlabel('radius (pixel)');
      
      %plot time vs radius
      figure;
      timePlot = Boxplots(this.timeArray, true);
      timePlot.setPosition(this.radiusArray);
      timePlot.plot();
      ylabel('time (s)');
      xlabel('radius (pixel)');
      xlim([0,this.radiusArray(end)+10]);
      
    end

  end
  
  methods (Access = protected)
    
    %METHOD: SETUP
    %Declare arrays for storing results
    function setup(this)
      this.meanArray = zeros(this.nRepeat, numel(this.radiusArray)); %mean of all pixels
      this.varianceArray = zeros(this.nRepeat, numel(this.radiusArray)); %variance of all pixels
      this.ksArray = zeros(this.nRepeat, numel(this.radiusArray)); %kolmogorov-smirnov p value
      %time to filter the image in seconds
      this.timeArray = zeros(this.nRepeat, numel(this.radiusArray));
    end
    
    %METHOD: DO EXPERIMENT
    %Filter Gaussian images for different radius multiple times
    function doExperiment(this)
      
      %for each radius
      for iRadius = 1:numel(this.radiusArray)
        
        %get the radius
        radius = this.radiusArray(iRadius);
        %instantiate an empirical null filter with that radius
        filter = EmpiricalNullFilter(radius);
        filter.setNInitial(this.nInitial);
        
        %for nRepeat times
        for iRepeat = 1:this.nRepeat
          
          %produce a gaussian image
          image = this.randStream.randn(256,256);
          
          %filter the image and time it
          tic;
          filter.filter(image);
          this.timeArray(iRepeat, iRadius) = toc;
          
          %get the filtered image and get the statistics
          image = reshape(filter.getFilteredImage(),[],1);
          this.meanArray(iRepeat, iRadius) = mean(image);
          this.varianceArray(iRepeat, iRadius) = var(image);
          [~, this.ksArray(iRepeat, iRadius)] = kstest(image);
          
          %print the progressbar
          this.printProgress( ((iRadius-1)*this.nRepeat + iRepeat) ... 
              / (numel(this.radiusArray) * this.nRepeat) );
          
        end
        
      end
      
    end
    
  end
  
end

