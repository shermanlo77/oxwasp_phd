%MIT License
%Copyright (c) 2019 Sherman Lo

%ABSTRCT CLASS: NULL ON IID DATA EXPERIMENT
%Investigate the null mean and null std on iid N(0,1) data
%Also investigates the (mean, std, kurtosis) moments of the normalised z statistics, that is let
    %Z~N(0,1), the normalised z statistics is (z - null mean) / null std.
%For a given n, n x N(0,1) are simulated, the null mean and null std are estimated using these
    %simulated data. The null parameters are recorded. The null parameters are then used normalise
    %the simulated data. The mean, variance and kurtosis of the normalised statistics are recorded.
    %This is repeated by simulating another data set.
%Various n are investigated
%METHODS TO BE IMPLEMENTED;
  %[nullMean, nullStd] = getNull(this, z)
    %given an array of data z, return the nullMean and nullStd
classdef (Abstract) NullIid < Experiment
  
  properties (SetAccess = protected)
    
    nArray = round(pi*(10:10:100).^2); %array of n to investigate
    nRepeat = 3; %number of times to repeat the experiment
    
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
      
      %plot null mean
      fig = LatexFigure.sub();
      boxplot = Boxplots(this.nullMeanArray);
      boxplot.setPosition(radiusPlot);
      boxplot.plot();
      hold on;
      zCritical = 2;
      plot(radiusPlot, zCritical./sqrt(this.nArray), 'k--');
      plot(radiusPlot, -zCritical./sqrt(this.nArray), 'k--');
      ax = fig.Children(1);
      ax.XLabel.String = 'r';
      ax.YLabel.String = 'null mean';
      ax.YLim = this.getYLim(1);
      ax.XLim(2) = 110;
      saveas(fig, fullfile(directory,strcat(this.experimentName,'_nullMean.eps')), 'epsc');
      
      %plot null std
      fig = LatexFigure.sub();
      boxplot = Boxplots(this.nullStdArray);
      boxplot.setPosition(radiusPlot);
      boxplot.plot();
      hold on;
      plot(radiusPlot, sqrt(chi2inv(normcdf(2),this.nArray - 1)./(this.nArray-1)), 'k--');
      plot(radiusPlot, sqrt(chi2inv(normcdf(-2),this.nArray - 1)./(this.nArray-1)), 'k--');
      ax = fig.Children(1);
      ax.XLabel.String = 'r';
      ax.YLabel.String = 'null std';
      ax.YLim = this.getYLim(2);
      ax.XLim(2) = 110;
      saveas(fig, fullfile(directory,strcat(this.experimentName,'_nullStd.eps')), 'epsc');
      
      %plot mean normalised z
      fig = LatexFigure.sub();
      boxplot = Boxplots(this.meanZArray);
      boxplot.setPosition(radiusPlot);
      boxplot.plot();
      hold on;
      zCritical = 2;
      plot(radiusPlot, zCritical./sqrt(this.nArray), 'k--');
      plot(radiusPlot, -zCritical./sqrt(this.nArray), 'k--');
      ax = fig.Children(1);
      ax.XLabel.String = 'r';
      ax.YLabel.String = 'mean corrected z';
      ax.YLim = this.getYLim(3);
      ax.XLim(2) = 110;
      saveas(fig, fullfile(directory,strcat(this.experimentName,'_zMean.eps')), 'epsc');
      
      %plot std normalised z
      fig = LatexFigure.sub();
      boxplot = Boxplots(this.stdZArray);
      boxplot.setPosition(radiusPlot);
      boxplot.plot();
      hold on;
      plot(radiusPlot, sqrt(chi2inv(normcdf(2),this.nArray - 1)./(this.nArray-1)), 'k--');
      plot(radiusPlot, sqrt(chi2inv(normcdf(-2),this.nArray - 1)./(this.nArray-1)), 'k--');
      ax = fig.Children(1);
      ax.XLabel.String = 'r';
      ax.YLabel.String = 'std corrected z';
      ax.YLim = this.getYLim(4);
      ax.XLim(2) = 110;
      saveas(fig, fullfile(directory,strcat(this.experimentName,'_zStd.eps')), 'epsc');
      
      %plot kurtosis normalised z
      fig = LatexFigure.sub();
      boxplot = Boxplots(this.kurtosisZArray);
      boxplot.setPosition(radiusPlot);
      boxplot.plot();
      hold on;
      plot(radiusPlot, 3+sqrt(24)*zCritical./sqrt(this.nArray), 'k--');
      plot(radiusPlot, 3-sqrt(24)*zCritical./sqrt(this.nArray), 'k--');
      ax = fig.Children(1);
      ax.XLabel.String = 'r';
      ax.YLabel.String = 'kurtosis corrected z';
      ax.YLim = this.getYLim(5);
      ax.XLim(2) = 110;
      saveas(fig, fullfile(directory,strcat(this.experimentName,'_zKurtosis.eps')), 'epsc');
      
    end
    
  end
  
  methods (Access = protected)
    
    %IMPLEMENTED: SETUP
    function setup(this, seed)
      this.randStream = RandStream('mt19937ar','Seed', seed);
      this.nullMeanArray = zeros(this.nRepeat, numel(this.nArray));
      this.nullStdArray = zeros(this.nRepeat, numel(this.nArray));
      this.meanZArray = zeros(this.nRepeat, numel(this.nArray));
      this.stdZArray = zeros(this.nRepeat, numel(this.nArray));
      this.kurtosisZArray = zeros(this.nRepeat, numel(this.nArray));
    end
    
    %IMPLEMENTED: DO EXPERIMENT
    function doExperiment(this)
      
      DebugPrint.newFile(this.experimentName);
      
      %for each n
      for iN = 1:numel(this.nArray)
        %get n, the number of N(0,1) samples to simulate
        n = this.nArray(iN);
        DebugPrint.write(strcat('n=',num2str(n)));
        
        %nRepeat times
        for iRepeat = 1:this.nRepeat
          
          z = this.getSample(n);
          %estimate the null mean and null std
          [nullMean, nullStd] = this.getNull(z);
          %save the null mean and null std
          this.nullMeanArray(iRepeat, iN) = nullMean;
          this.nullStdArray(iRepeat, iN) = nullStd;
          
          %save the mean, variance and kurtosis of the normalised z statistics
          z = (z - nullMean) / nullStd;
          this.meanZArray(iRepeat, iN) = mean(z);
          this.stdZArray(iRepeat, iN) = std(z);
          this.kurtosisZArray(iRepeat, iN) = kurtosis(z);
          
          this.printProgress( ((iN-1)*this.nRepeat + iRepeat) / (numel(this.nArray)*this.nRepeat) );
          
        end
        
      end
      
      DebugPrint.close();

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
    
  end
  
  methods (Abstract, Access = protected)
    
    %ABSTRACT METHOD: GET NULL
    %given an array of data z, return the nullMean and nullStd
    [nullMean, nullStd] = getNull(this, z);
    
  end
  
end

