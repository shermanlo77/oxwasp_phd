%ABSTRACT CLASS: EXPERIMENT ALL GAUSSIAN
%See how the empirical null filter behaves for images of pure Gaussian noise
%A image is produced and then filtered using the empirical null filter. Various properties of the
%  post-filtered image are recorded such as the mean, variance, p-value from the KS test and the
%  time it took to filter the image.
%This is repeated nRepeat times for various kernel radius
%Plots the following:
%  post-filter mean vs radius
%  post-filter variance vs radius
%  log ks p value vs radius
%  time to filter vs radius
%
%Methods to be implemeted:
%  getImage returns an image to be filtered, this would be a Gaussian image with some bias added to
%  or mutiplied to it
classdef Experiment_AllNull < Experiment
  
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
    
    randStream; %rng
    
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    function this = Experiment_AllNull(experimentName)
      this@Experiment(experimentName);
    end
    
    %METHOD: PRINT RESULTS
    function printResults(this)
      
      %where to save the figures
      directory = fullfile('reports','figures','inference','allnull');
      
      %save properties of this experiment to txt
      
      %radius range
      fildId = fopen(fullfile(directory,strcat(this.experiment_name,'radius1.txt')),'w');
      fprintf(fildId,'%d',this.radiusArray(1));
      fclose(fildId);
      
      %radius range
      fildId = fopen(fullfile(directory,strcat(this.experiment_name,'radiusend.txt')),'w');
      fprintf(fildId,'%d',this.radiusArray(end));
      fclose(fildId);
      
      %nrepeat
      fildId = fopen(fullfile(directory,strcat(this.experiment_name,'nrepeat.txt')),'w');
      fprintf(fildId,'%d',this.nRepeat);
      fclose(fildId);
      
      %imagesize
      fildId = fopen(fullfile(directory,strcat(this.experiment_name,'height.txt')),'w');
      fprintf(fildId,'%d',this.imageSize(1));
      fclose(fildId);
      
      %imagesize
      fildId = fopen(fullfile(directory,strcat(this.experiment_name,'width.txt')),'w');
      fprintf(fildId,'%d',this.imageSize(2));
      fclose(fildId);
      
      %show critical region for mean and variance
      sigma = 2; %get sigma level
      alpha = 2*(1-normcdf(sigma)); %covert to significant level
      n = this.imageSize(1) * this.imageSize(2); %get number of pixels in the image
      
      %plot post filter mean vs radius
      fig = LatexFigure.sub();
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
      saveas(fig,fullfile(directory, strcat(this.experiment_name,'mean.eps')),'epsc');
      
      %plot post filter variance vs radius
      fig = LatexFigure.sub();
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
      saveas(fig,fullfile(directory, strcat(this.experiment_name,'variance.eps')),'epsc');
      
      %plot time vs radius
      fig = LatexFigure.sub();
      timePlot = Boxplots(this.timeArray, true);
      timePlot.setPosition(this.radiusArray);
      timePlot.plot();
      ylabel('time (s)');
      xlabel('radius (pixel)');
      xlim([0,this.radiusArray(end)+10]);
      saveas(fig,fullfile(directory, strcat(this.experiment_name,'time.eps')),'epsc');
      
    end

  end
  
  methods (Access = protected)
    
    %METHOD: SETUP
    %Declare arrays for storing results
    function setup(this, seed)
      this.meanArray = zeros(this.nRepeat, numel(this.radiusArray)); %mean of all pixels
      this.varianceArray = zeros(this.nRepeat, numel(this.radiusArray)); %variance of all pixels
      this.ksArray = zeros(this.nRepeat, numel(this.radiusArray)); %kolmogorov-smirnov p value
      %time to filter the image in seconds
      this.timeArray = zeros(this.nRepeat, numel(this.radiusArray));
      %set the rng
      this.randStream = RandStream('mt19937ar','Seed', seed);
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
          image = this.getImage();
          
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
  
  methods (Abstract, Access = protected)
    
    %ABSTRACT METHOD:
    %Return the method to filter using the rng
    image = getImage(this)
  end
  
end

