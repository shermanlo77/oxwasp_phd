classdef Experiment_AllGaussian < Experiment
  
  properties (SetAccess = private)
    
    %array of kernel radius to investigate
    radiusArray = 10:10:100;
    %number of times to repeat the experiment
    nRepeat = 100;
    
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
    
    function printResults(this)
      
      sigma = 2;
      alpha = 2*(1-normcdf(sigma));
      n = this.imageSize(1) * this.imageSize(2);
      
      figure;
      meanPlot = Boxplots(this.meanArray(:,1:5), true);
      meanPlot.setPosition(this.radiusArray(1:5));
      meanPlot.plot();
      hold on;
      meanCritical = sigma/sqrt(n);
      plot([0,this.radiusArray(end)],[meanCritical,meanCritical], 'k--');
      plot([0,this.radiusArray(end)],[-meanCritical,-meanCritical], 'k--');
      ylabel('post filter image greyvalue mean');
      xlabel('radius (pixel)');
      
      figure;
      varPlot = Boxplots(this.varianceArray(:, 1:5), true);
      varPlot.setPosition(this.radiusArray(1:5));
      varPlot.plot();
      hold on;
      varCritical1 = chi2inv(alpha, n-1)/(n-1);
      varCritical2 = chi2inv(1-alpha, n-1)/(n-1);
      plot([0,this.radiusArray(end)],[varCritical1,varCritical1], 'k--');
      plot([0,this.radiusArray(end)],[varCritical2,varCritical2], 'k--');
      ylabel('post filter image greyvalue variance');
      xlabel('radius (pixel)');
      
      
      fig = figure;
      ksPlot = Boxplots(this.ksArray(:, 1:5), true);
      ksPlot.setPosition(this.radiusArray(1:5));
      ksPlot.plot();
      hold on;
      plot([0,this.radiusArray(end)],[alpha,alpha], 'k--');
      fig.Children(1).YLim(1) = 0;
      ylabel('KS p-value');
      xlabel('radius (pixel)');
      
      figure;
      timePlot = Boxplots(this.timeArray(:, 1:5), true);
      timePlot.setPosition(this.radiusArray(1:5));
      timePlot.plot();
      ylabel('time (s)');
      xlabel('radius (pixel)')
      
    end

  end
  
  methods (Access = protected)
    
    function setup(this)
      this.meanArray = zeros(this.nRepeat, numel(this.radiusArray)); %mean of all pixels
      this.varianceArray = zeros(this.nRepeat, numel(this.radiusArray)); %variance of all pixels
      this.ksArray = zeros(this.nRepeat, numel(this.radiusArray)); %kolmogorov-smirnov p value
      %time to filter the image in seconds
      this.timeArray = zeros(this.nRepeat, numel(this.radiusArray));
    end
    
    function doExperiment(this)
      
      %for each radius
      for iRadius = 1:numel(this.radiusArray)
        
        %get the radius
        radius = this.radiusArray(iRadius);
        %instantiate an empirical null filter with that radius
        filter = EmpiricalNullFilter(radius);
        
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

