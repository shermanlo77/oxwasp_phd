%MIT License
%Copyright (c) 2019 Sherman Lo

%ABSTRACT CLASS: ALL NULL IMAGE EXPERIMENT
%See how the null filters behaves for images which has no defects
%
%An image is produced and then filtered using a null filter. Various properties of the post-filtered
    %image are recorded such as the mean, variance, kurtosis and the time it took to filter the
    %image.
%This is repeated nRepeat times for various kernel radius
%Plots the following:
%  post-filter mean vs radius
%  post-filter variance vs radius
%  post-filter kurtosis vs radius
%  time to filter vs radius
%
%Methods to be implemeted:
%  getImage returns an image to be filtered, this would be a Gaussian image with some bias added to
%  or mutiplied to it, ie contamination
classdef AllNull < Experiment
  
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
    meanArray; %mean over all pixels
    stdArray; %variance over all pixels
    kurtosisArray; %kurtosis over all pixels
    timeArray; %time to filter the image in seconds
    
    imageSize = [256, 256]; %size of the gaussian image
    
    randStream; %rng
    
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    function this = AllNull()
      this@Experiment();
    end
    
    %IMPLEMENTED: PRINT RESULTS
    function printResults(this)
      
      %where to save the figures
      directory = fullfile('reports','figures','inference');
      
      %show critical region for mean and variance
      alpha = 2*(1-normcdf(2)); %significant level
      n = this.imageSize(1) * this.imageSize(2); %get number of pixels in the image
      
      %plot post filter mean vs radius
      fig = LatexFigure.sub();
      meanPlot = Boxplots(this.meanArray);
      meanPlot.setPosition(this.radiusArray);
      meanPlot.plot();
      hold on;
      meanCritical = norminv(1-alpha/2)/sqrt(n);
      plot([0,this.radiusArray(end)+10],[meanCritical,meanCritical], 'k--');
      plot([0,this.radiusArray(end)+10],[-meanCritical,-meanCritical], 'k--');
      xlim([0,this.radiusArray(end)+10]);
      ylabel('post filter image greyvalue mean');
      xlabel('radius (px)');
      if (~isempty(this.getYLim(3)))
        ylim(this.getYLim(3));
      end
      saveas(fig,fullfile(directory, strcat(this.experimentName,'_mean.eps')),'epsc');
      
      %plot post filter variance vs radius
      fig = LatexFigure.sub();
      stdPlot = Boxplots(this.stdArray);
      stdPlot.setPosition(this.radiusArray);
      stdPlot.plot();
      hold on;
      stdCritical1 = sqrt(chi2inv(alpha/2, n-1)/(n-1));
      stdCritical2 = sqrt(chi2inv(1-alpha/2, n-1)/(n-1));
      plot([0,this.radiusArray(end)+10],[stdCritical1,stdCritical1], 'k--');
      plot([0,this.radiusArray(end)+10],[stdCritical2,stdCritical2], 'k--');
      xlim([0,this.radiusArray(end)+10]);
      ylabel('post filter image greyvalue std');
      xlabel('radius (px)');
      if (~isempty(this.getYLim(4)))
        ylim(this.getYLim(4));
      end
      saveas(fig,fullfile(directory, strcat(this.experimentName,'_variance.eps')),'epsc');
      
      %plot post filter kurtosisArray vs radius
      fig = LatexFigure.sub();
      kurtPlot = Boxplots(this.kurtosisArray);
      kurtPlot.setPosition(this.radiusArray);
      kurtPlot.plot();
      hold on;
      meanCritical = norminv(1-alpha/2)/sqrt(n);
      plot([0,this.radiusArray(end)+10],3+sqrt(24)*[meanCritical,meanCritical], 'k--');
      plot([0,this.radiusArray(end)+10],3+sqrt(24)*[-meanCritical,-meanCritical], 'k--');
      xlim([0,this.radiusArray(end)+10]);
      ylabel('post filter image greyvalue kurtosis');
      xlabel('radius (px)');
      if (~isempty(this.getYLim(5)))
        ylim(this.getYLim(5));
      end
      saveas(fig,fullfile(directory, strcat(this.experimentName,'_kurtosis.eps')),'epsc');
      
      %plot time vs radius
      fig = LatexFigure.sub();
      timePlot = Boxplots(this.timeArray);
      timePlot.setPosition(this.radiusArray);
      timePlot.plot();
      ylabel('time (s)');
      xlabel('radius (px)');
      xlim([0,this.radiusArray(end)+10]);
      if (~isempty(this.getYLim(6)))
        ylim(this.getYLim(6));
      end
      saveas(fig,fullfile(directory, strcat(this.experimentName,'_time.eps')),'epsc');
      
    end
    
    %METHOD: PRINT TIME
    %Print the mean and standard deviation of the running time for a given kernel radius
    function printTime(this, iRadius)
      time = mean(this.timeArray(:,iRadius));
      timeError = std(this.timeArray(:,iRadius));
      disp(cell2mat({'running time for radius = ',num2str(this.radiusArray(iRadius)), ...
          ' in seconds: ', num2str(time), '±',num2str(timeError)}));
    end

  end
  
  methods (Access = protected)
    
    %IMPLEMENTED: SETUP
    %Declare arrays for storing results
    function setup(this, seed)
      this.meanArray = zeros(this.nRepeat, numel(this.radiusArray));
      this.stdArray = zeros(this.nRepeat, numel(this.radiusArray));
      this.kurtosisArray = zeros(this.nRepeat, numel(this.radiusArray));
      %time to filter the image in seconds
      this.timeArray = zeros(this.nRepeat, numel(this.radiusArray));
      %set the rng
      this.randStream = RandStream('mt19937ar','Seed', seed);
    end
    
    %IMPLEMENTED: DO EXPERIMENT
    %Filter Gaussian images for different radius multiple times
    function doExperiment(this)
      
      %for each radius
      for iRadius = 1:numel(this.radiusArray)
        
        %get the radius
        radius = this.radiusArray(iRadius);
        %instantiate a null filter with that radius
        filter = this.getFilter(radius);
        filter.setNInitial(this.nInitial);
        
        %for nRepeat times
        for iRepeat = 1:this.nRepeat
          
          %produce a gaussian image
          image = this.getImage();
          
          %filter the image and time it
          tic;
          filter.filter(image);
          this.timeArray(iRepeat, iRadius) = toc;
          
          %get the filtered image
          image = filter.getFilteredImage();
          
          %save the normalised statistics mean, var and kurtosis
          image = reshape(image,[],1);
          this.meanArray(iRepeat, iRadius) = mean(image);
          this.stdArray(iRepeat, iRadius) = std(image);
          this.kurtosisArray(iRepeat, iRadius) = kurtosis(image);
          
          %print the progressbar
          this.printProgress( ((iRadius-1)*this.nRepeat + iRepeat) ... 
              / (numel(this.radiusArray) * this.nRepeat) );
          
        end
        
      end
      
    end
    
  end
  
  methods (Abstract, Access = protected)
    
    %ABSTRACT METHOD: GET FILTER
    %Return an instantiated a null filter with that radius
    filter = getFilter(this, radius)
    
    %ABSTRACT METHOD: GET IMAGE
    %Return an image to filter using its rng
    image = getImage(this)
    
    %ABSTRACT METHOD: GET YLIM
    %Return the ylim for each graph in printResults
    ylim = getYLim(this, graphIndex)
  end
  
end

