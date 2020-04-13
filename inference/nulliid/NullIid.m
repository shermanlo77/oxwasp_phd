%MIT License
%Copyright (c) 2019 Sherman Lo

%CLASS: NULL ON IID DATA EXPERIMENT
%Investigate different mean and std estimators for generated data
%Various n are investigated and repeated
%Plot sampling distributions as box plot
classdef NullIid < Experiment
  
  properties (SetAccess = protected)
    
    nArray = round(pi*(10:10:100).^2); %array of n to investigate
    nRepeat = 100; %number of times to repeat the experiment
    nEstimator = 4;
    
    %array of results
      %dim 1: for each repeat
      %dim 2: for each n
    nullMeanArray; %null mean
    nullStdArray; %null std
    meanZArray; %mean normalised z
    stdZArray; %std normalised z
    kurtosisZArray; %kurtosis normalised z
    
    randStream; %rng
    
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    function this = NullIid()
      this@Experiment();
    end
    
    %IMPLEMENTED: PRINT RESULTS
    %Plots the null mean and null variance for different n
    %Plots the mean, variance and kurtosis of the normalised z statistics for different n 
    function printResults(this)
      
      directory = fullfile('reports','figures','inference');
      
      radiusPlot = sqrt(this.nArray/pi); %treat n as as circular kernel area
      
      for iEstimator = 1:this.nEstimator
      
        %plot null mean
        fig = LatexFigure.sub();
        boxplot = Boxplots(this.nullMeanArray(:,:,iEstimator));
        boxplot.setPosition(radiusPlot);
        boxplot.plot();
        hold on;
        zCritical = 2;
        plot(radiusPlot, zCritical./sqrt(this.nArray), 'k--');
        plot(radiusPlot, -zCritical./sqrt(this.nArray), 'k--');
        ax = fig.Children(1);
        ax.XLabel.String = 'r';
        ax.YLabel.String = 'central tendency';
        ax.YLim = this.getYLim(1);
        ax.XLim(2) = 110;
        ax.Box = 'on';
        saveas(fig, fullfile(directory,strcat(this.experimentName,'_nullMean', ...
            num2str(iEstimator), '.eps')), 'epsc');

        %plot null std
        fig = LatexFigure.sub();
        boxplot = Boxplots(this.nullStdArray(:,:,iEstimator));
        boxplot.setPosition(radiusPlot);
        boxplot.plot();
        hold on;
        plot(radiusPlot, sqrt(chi2inv(normcdf(2),this.nArray - 1)./(this.nArray-1)), 'k--');
        plot(radiusPlot, sqrt(chi2inv(normcdf(-2),this.nArray - 1)./(this.nArray-1)), 'k--');
        ax = fig.Children(1);
        ax.XLabel.String = 'r';
        ax.YLabel.String = 'dispersion';
        ax.YLim = this.getYLim(2);
        ax.XLim(2) = 110;
        ax.Box = 'on';
        saveas(fig, fullfile(directory,strcat(this.experimentName,'_nullStd', ...
            num2str(iEstimator), '.eps')), 'epsc');
      
      end
      
    end
    
  end
  
  methods (Access = protected)
    
    %IMPLEMENTED: SETUP
    function setup(this)
      this.randStream = RandStream('mt19937ar','Seed', uint32(2288468478));
      this.nullMeanArray = zeros(this.nRepeat, numel(this.nArray), 4);
      this.nullStdArray = zeros(this.nRepeat, numel(this.nArray), 4);
      this.meanZArray = zeros(this.nRepeat, numel(this.nArray), 4);
      this.stdZArray = zeros(this.nRepeat, numel(this.nArray), 4);
      this.kurtosisZArray = zeros(this.nRepeat, numel(this.nArray), 4);
    end
    
    %IMPLEMENTED: DO EXPERIMENT
    function doExperiment(this)
      
      this.setNIteration(this.nEstimator*numel(this.nArray)*this.nRepeat);
      
      %for each estimator
      for iEstimator = 1:this.nEstimator
      
        %for each n
        for iN = 1:numel(this.nArray)
          %get n, the number of N(0,1) samples to simulate
          n = this.nArray(iN);

          %nRepeat times
          for iRepeat = 1:this.nRepeat

            z = this.getSample(n);
            %estimate the null mean and null std
            [nullMean, nullStd] = this.getNull(z, iEstimator);
            %save the null mean and null std
            this.nullMeanArray(iRepeat, iN, iEstimator) = nullMean;
            this.nullStdArray(iRepeat, iN, iEstimator) = nullStd;

            this.madeProgress();

          end

        end
        
      end

    end
    
  end
  
  methods (Access = protected)
    
    %METHODS: GET SAMPLE
    %Return n N(0,1) data
    %May be overriden in subclasses
    %PARAMETERS:
      %n: number of data to simulate
    function z = getSample(this, n)
      z = this.randStream.randn(n,1);
    end
    
    %METHOD: GET YLIM
    %Return the ylim for each of the graphs in printResults
    function yLim = getYLim(this, index)
      switch index
        case 1
          yLim = [-0.4, 0.4];
        case 2
          yLim = [0.7, 1.6];
        case 3
          yLim = [-0.4, 0.4];
        case 4
          yLim = [0.6, 1.3];
        case 5
          yLim = [2.4, 4.0];
      end
    end
    
    
    %METHOD: GET NULL
    %given an array of data z and estimator, return the nullMean and nullStd
    function [nullMean, nullStd] = getNull(this, z, iEstimator)
      switch iEstimator
        case 1
          %empirical null mean and std
          empiricalNull = EmpiricalNull(z, 0, ...
          this.randStream.randi([intmin('int32'),intmax('int32')],'int32'));
          empiricalNull.estimateNull();
          nullMean = empiricalNull.getNullMean();
          nullStd = empiricalNull.getNullStd();
        case 2
          %robust bisqure mean
          %MAD
          empiricalNull = MadModeNull(z, 0, ...
          this.randStream.randi([intmin('int32'),intmax('int32')],'int32'));
          empiricalNull.estimateNull();
          nullMean = robustfit(ones(numel(z), 1), z, [], [], 'off');
          nullStd = empiricalNull.getNullStd();
        case 3
          nullMean = mean(z);
          nullStd = std(z);
        case 4
          q = quantile(z, [0.25, 0.5, 0.75]);
          nullMean = q(2);
          nullStd = (q(3)-q(1))/1.349;
      end
      
    end
    
  end
  
end

