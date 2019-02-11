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
    meanArray; %mean of all pixels
    stdArray; %variance of all pixels
    kurtosisArray; %kurtosis of all pixels
    timeArray; %time to filter the image in seconds
    
    %array to store the corrected statistics, empirical null mean and empirical null std images,
        %one for each kernel radius
      %dim 1: y axis of image
      %dim 2: x axis of image
      %dim 3: for each kernel radius
    correctedZArray;
    nullMeanArray;
    nullStdArray;
    
    imageSize = [256, 256]; %size of the gaussian image
    
    randStream; %rng
    
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    function this = AllNull()
      this@Experiment();
    end
    
    %METHOD: PRINT RESULTS
    function printResults(this)
      
      %where to save the figures
      directory = fullfile('reports','figures','inference','allnull');
      
      %save properties of this experiment to txt
      
      %radius range
      fildId = fopen(fullfile(directory,strcat(this.experiment_name,'_radius1.txt')),'w');
      fprintf(fildId,'%d',this.radiusArray(1));
      fclose(fildId);
      
      %radius range
      fildId = fopen(fullfile(directory,strcat(this.experiment_name,'_radiusend.txt')),'w');
      fprintf(fildId,'%d',this.radiusArray(end));
      fclose(fildId);
      
      %nrepeat
      fildId = fopen(fullfile(directory,strcat(this.experiment_name,'_nrepeat.txt')),'w');
      fprintf(fildId,'%d',this.nRepeat);
      fclose(fildId);
      
      %imagesize
      fildId = fopen(fullfile(directory,strcat(this.experiment_name,'_height.txt')),'w');
      fprintf(fildId,'%d',this.imageSize(1));
      fclose(fildId);
      
      %imagesize
      fildId = fopen(fullfile(directory,strcat(this.experiment_name,'_width.txt')),'w');
      fprintf(fildId,'%d',this.imageSize(2));
      fclose(fildId);
      
      %show critical region for mean and variance
      alpha = 0.05; %significant level
      n = this.imageSize(1) * this.imageSize(2); %get number of pixels in the image
      
      %plot empirical null mean
      fig = LatexFigure.sub();
      nullMeanPlot = Boxplots(reshape(this.nullMeanArray,[],numel(this.radiusArray)));
      nullMeanPlot.setPosition(this.radiusArray);
      nullMeanPlot.setWantOutlier(false);
      nullMeanPlot.plot();
      hold on;
      A = pi*this.radiusArray.^2;
      plot(this.radiusArray, norminv(1-alpha/2)./sqrt(A), 'k--');
      plot(this.radiusArray, -norminv(1-alpha/2)./sqrt(A), 'k--');
      ylabel('empirical null mean');
      xlabel('radius (pixel)');
      saveas(fig,fullfile(directory, strcat(this.experiment_name,'_nullMean.eps')),'epsc');
      
      %plot empirical null var
      fig = LatexFigure.sub();
      nullVarPlot = Boxplots(reshape(this.nullStdArray,[],numel(this.radiusArray)));
      nullVarPlot.setPosition(this.radiusArray);
      nullVarPlot.setWantOutlier(false);
      nullVarPlot.plot();
      hold on;
      A = pi*this.radiusArray.^2;
      plot(this.radiusArray, sqrt(chi2inv(alpha/2, A-1)./(A-1)), 'k--');
      plot(this.radiusArray, sqrt(chi2inv(1-alpha/2, A-1)./(A-1)), 'k--');
      ylabel('empirical null std');
      xlabel('radius (pixel)');
      saveas(fig,fullfile(directory, strcat(this.experiment_name,'_nullStd.eps')),'epsc');
      
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
      xlabel('radius (pixel)');
      saveas(fig,fullfile(directory, strcat(this.experiment_name,'_mean.eps')),'epsc');
      
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
      xlabel('radius (pixel)');
      saveas(fig,fullfile(directory, strcat(this.experiment_name,'_variance.eps')),'epsc');
      
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
      xlabel('radius (pixel)');
      saveas(fig,fullfile(directory, strcat(this.experiment_name,'_kurtosis.eps')),'epsc');
      
      %plot time vs radius
      fig = LatexFigure.sub();
      timePlot = Boxplots(this.timeArray);
      timePlot.setPosition(this.radiusArray);
      timePlot.plot();
      ylabel('time (s)');
      xlabel('radius (pixel)');
      xlim([0,this.radiusArray(end)+10]);
      saveas(fig,fullfile(directory, strcat(this.experiment_name,'_time.eps')),'epsc');
      
    end

  end
  
  methods (Access = protected)
    
    %METHOD: SETUP
    %Declare arrays for storing results
    function setup(this, seed)
      this.meanArray = zeros(this.nRepeat, numel(this.radiusArray)); %mean of all pixels
      this.stdArray = zeros(this.nRepeat, numel(this.radiusArray)); %variance of all pixels
      this.kurtosisArray = zeros(this.nRepeat, numel(this.radiusArray)); %kolmogorov-smirnov p value
      %time to filter the image in seconds
      this.timeArray = zeros(this.nRepeat, numel(this.radiusArray));
      this.correctedZArray = zeros(this.imageSize(1), this.imageSize(2), numel(this.radiusArray));
      this.nullMeanArray = zeros(this.imageSize(1), this.imageSize(2), numel(this.radiusArray));
      this.nullStdArray = zeros(this.imageSize(1), this.imageSize(2), numel(this.radiusArray));
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
        filter = this.getFilter(radius);
        filter.setNInitial(this.nInitial);
        filter.setSeed(this.randStream.randi([intmin('int32'),intmax('int32')],'int32'));
        
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
          
          %for the first repeat, save the corrected image, null mean and null std
          if (iRepeat == 1)
            this.nullMeanArray(:,:,iRadius) = filter.getNullMean();
            this.nullStdArray(:,:,iRadius) = filter.getNullStd();
            this.correctedZArray(:,:,iRadius) = image;
          end
          
          %save the corrected statistics mean, var and kurtosis
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
    
    %METHOD: GET FILTER
    %instantiate an empirical null filter with that radius
    function filter = getFilter(this, radius)
      filter = EmpiricalNullFilter(radius);
    end
    
  end
  
  methods (Abstract, Access = protected)
    
    %ABSTRACT METHOD:
    %Return the method to filter using the rng
    image = getImage(this)
  end
  
end
