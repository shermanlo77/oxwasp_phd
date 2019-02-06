%CLASS: EXPERIMENT EMPIRICAL NULL ON IID
%Investigate the empirical null mean and empirical null std on iid N(0,1) data
%Also investigates the moments of the corrected z statistics, that is let Z~N(0,1), the corrected z
    %statistics is (z - empirical null mean) / empirical null std
%For a given n, n x N(0,1) are simulated, the empirical null is then conducted on these simulated
    %data. The empirical null parameters are recorded. The empirical null parameters are then used
    %to correct the simulated data, the mean, variance and kurtosis are recorded
classdef (Abstract) NullIid < Experiment
  
  properties (SetAccess = protected)
    
    nArray = round(pi*(10:10:100).^2); %array of n to investigate
    nRepeat = 100; %number of times to repeat the experiment
    
    %array of results
      %dim 1: for each repeat
      %dim 2: for each n
    nullMeanArray; %empirical null mean
    nullStdArray; %empirical null std
    meanZArray; %mean corrected z
    stdZArray; %std corrected z
    kurtosisZArray; %kurtosis corrected z
    
    randStream; %rng
    
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    function this = NullIid()
      this@Experiment();
    end
    
    %IMPLEMENTED: PRINT RESULTS
    %Plots the empirical null mean and empirical null variance for different n
    %Plots the mean, variance and kurtosis of the corrected z statistics for different n 
    function printResults(this)
      
      directory = fullfile('reports','figures','inference','NullIid');
      
      radiusPlot = sqrt(this.nArray/pi);
      
      %plot empirical null mean
      fig = LatexFigure.sub();
      boxplot = Boxplots(this.nullMeanArray);
      boxplot.setPosition(radiusPlot);
      %boxplot.setWantOutlier(false);
      boxplot.plot();
      hold on;
      zCritical = norminv(0.975);
      plot(radiusPlot, zCritical./sqrt(this.nArray), 'k--');
      plot(radiusPlot, -zCritical./sqrt(this.nArray), 'k--');
      ax = fig.Children(1);
      ax.XLabel.String = 'r';
      ax.YLabel.String = 'empirical null mean';
      ax.YLim = this.getYLim(1);
      saveas(fig, fullfile(directory,strcat(this.experiment_name,'_nullMean.eps')), 'epsc');
      
      %plot empirical null std
      fig = LatexFigure.sub();
      boxplot = Boxplots(this.nullStdArray);
      boxplot.setPosition(radiusPlot);
      %boxplot.setWantOutlier(false);
      boxplot.plot();
      hold on;
      plot(radiusPlot, sqrt(chi2inv(0.975,this.nArray - 1)./(this.nArray-1)), 'k--');
      plot(radiusPlot, sqrt(chi2inv(0.025,this.nArray - 1)./(this.nArray-1)), 'k--');
      ax = fig.Children(1);
      ax.XLabel.String = 'r';
      ax.YLabel.String = 'empirical null std';
      ax.YLim = this.getYLim(2);
      saveas(fig, fullfile(directory,strcat(this.experiment_name,'_nullStd.eps')), 'epsc');
      
      %plot mean corrected z
      fig = LatexFigure.sub();
      boxplot = Boxplots(this.meanZArray);
      boxplot.setPosition(radiusPlot);
      %boxplot.setWantOutlier(false);
      boxplot.plot();
      hold on;
      zCritical = norminv(0.975);
      plot(radiusPlot, zCritical./sqrt(this.nArray), 'k--');
      plot(radiusPlot, -zCritical./sqrt(this.nArray), 'k--');
      ax = fig.Children(1);
      ax.XLabel.String = 'r';
      ax.YLabel.String = 'mean corrected z';
      ax.YLim = this.getYLim(3);
      saveas(fig, fullfile(directory,strcat(this.experiment_name,'_zMean.eps')), 'epsc');
      
      %plot std corrected z
      fig = LatexFigure.sub();
      boxplot = Boxplots(this.stdZArray);
      boxplot.setPosition(radiusPlot);
      %boxplot.setWantOutlier(false);
      boxplot.plot();
      hold on;
      plot(radiusPlot, sqrt(chi2inv(0.975,this.nArray - 1)./(this.nArray-1)), 'k--');
      plot(radiusPlot, sqrt(chi2inv(0.025,this.nArray - 1)./(this.nArray-1)), 'k--');
      ax = fig.Children(1);
      ax.XLabel.String = 'r';
      ax.YLabel.String = 'std corrected z';
      ax.YLim = this.getYLim(4);
      saveas(fig, fullfile(directory,strcat(this.experiment_name,'_zStd.eps')), 'epsc');
      
      %plot kurtosis corrected z
      fig = LatexFigure.sub();
      boxplot = Boxplots(this.kurtosisZArray);
      boxplot.setPosition(radiusPlot);
      %boxplot.setWantOutlier(false);
      boxplot.plot();
      hold on;
      plot(radiusPlot, 3+sqrt(24)*zCritical./sqrt(this.nArray), 'k--');
      plot(radiusPlot, 3-sqrt(24)*zCritical./sqrt(this.nArray), 'k--');
      ax = fig.Children(1);
      ax.XLabel.String = 'r';
      ax.YLabel.String = 'kurtosis corrected z';
      ax.YLim = this.getYLim(5);
      saveas(fig, fullfile(directory,strcat(this.experiment_name,'_zKurtosis.eps')), 'epsc');
      
    end
    
  end
  
  methods (Access = protected)
    
    %IMPLEMENTED: SETUP
    function setup(this, seed)
      this.randStream = RandStream('mt19937ar','Seed', uint32(seed));
      this.nullMeanArray = zeros(this.nRepeat, numel(this.nArray));
      this.nullStdArray = zeros(this.nRepeat, numel(this.nArray));
      this.meanZArray = zeros(this.nRepeat, numel(this.nArray));
      this.stdZArray = zeros(this.nRepeat, numel(this.nArray));
      this.kurtosisZArray = zeros(this.nRepeat, numel(this.nArray));
    end
    
    %IMPLEMENTED: DO EXPERIMENT
    function doExperiment(this)
      
      %for each n
      for iN = 1:numel(this.nArray)
        %get n, the number of N(0,1) samples to simulate
        n = this.nArray(iN);
        
        %nRepeat times
        for iRepeat = 1:this.nRepeat
          
          z = this.getSample(n);
          %instantiate empirical null, get the empirical null
          [nullMean, nullStd] = this.getNull(z);
          this.nullMeanArray(iRepeat, iN) = nullMean;
          this.nullStdArray(iRepeat, iN) = nullStd;
          
          %save the mean, variance and kurtosis of the corrected z statistics
          this.meanZArray(iRepeat, iN) = mean(z);
          this.stdZArray(iRepeat, iN) = std(z);
          this.kurtosisZArray(iRepeat, iN) = kurtosis(z);
          
          this.printProgress( ((iN-1)*this.nRepeat + iRepeat) / (numel(this.nArray)*this.nRepeat) );
          
        end
        
      end

    end
    
  end
  
  methods (Access = protected)
    
    function z = getSample(this, n)
      z = this.randStream.randn(n,1); %simulate N(0,1);
    end
    
    function yLim = getYLim(this, index)
      switch index
        case 1
          yLim = [-0.4, 0.4];
        case 2
          yLim = [0.7, 1.8];
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
      
    [nullMean, nullStd] = getNull(this, z);
      
  end
  
end

