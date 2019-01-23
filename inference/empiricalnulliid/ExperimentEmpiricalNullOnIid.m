%CLASS: EXPERIMENT EMPIRICAL NULL ON IID
%Investigate the empirical null mean and empirical null std on iid N(0,1) data
%Also investigates the moments of the corrected z statistics, that is let Z~N(0,1), the corrected z
    %statistics is (z - empirical null mean) / empirical null std
%For a given n, n x N(0,1) are simulated, the empirical null is then conducted on these simulated
    %data. The empirical null parameters are recorded. The empirical null parameters are then used
    %to correct the simulated data, the mean, variance and kurtosis are recorded
classdef ExperimentEmpiricalNullOnIid < Experiment
  
  properties (SetAccess = protected)
    
    nArray = round(10.^linspace(2,5,10)); %array of n to investigate
    nRepeat = 256*256; %number of times to repeat the experiment
    
    %array of results
      %dim 1: for each repeat
      %dim 2: for each n
    nullMeanArray; %empirical null mean
    nullStdArray; %empirical null std
    meanZArray; %mean corrected z
    varZArray; %variance corrected z
    kurtosisZArray; %kurtosis corrected z
    
    %array of sample of corrected z (cell array)
      %dim 1: for each n
      %each element contains a n-size column vector
    correctedZArray;
    
    randStream; %rng
    
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    function this = ExperimentEmpiricalNullOnIid()
      this@Experiment('ExperimentEmpiricalNullOnIid');
    end
    
    %IMPLEMENTED: PRINT RESULTS
    %Plots the empirical null mean and empirical null variance for different n
    %Plots the mean, variance and kurtosis of the corrected z statistics for different n 
    function printResults(this)
      
      %plot empirical null mean
      fig = figure;
      boxplot = Boxplots(this.nullMeanArray, true);
      boxplot.setPosition(this.nArray);
      boxplot.plot();
      hold on;
      zCritical = norminv(0.975);
      plot(this.nArray, zCritical./sqrt(this.nArray), 'k--');
      plot(this.nArray, -zCritical./sqrt(this.nArray), 'k--');
      ax = fig.Children(1);
      ax.XScale = 'log';
      ax.XLabel.String = 'n';
      ax.YLabel.String = 'empirical null mean';
      ax.XLim(1) = ax.XLim(1)/10^(0.1);
      ax.XLim(2) = ax.XLim(2)*10^(0.1);
      
      %plot empirical null var
      fig = figure;
      boxplot = Boxplots(this.nullStdArray.^2, true);
      boxplot.setPosition(this.nArray);
      boxplot.plot();
      hold on;
      plot(this.nArray, chi2inv(0.975,this.nArray - 1)./(this.nArray-1), 'k--');
      plot(this.nArray, chi2inv(0.025,this.nArray - 1)./(this.nArray-1), 'k--');
      ax = fig.Children(1);
      ax.XScale = 'log';
      ax.XLabel.String = 'n';
      ax.YLabel.String = 'empirical null variance';
      ax.XLim(1) = ax.XLim(1)/10^(0.1);
      ax.XLim(2) = ax.XLim(2)*10^(0.1);
      
      %plot mean corrected z
      fig = figure;
      boxplot = Boxplots(this.meanZArray, true);
      boxplot.setPosition(this.nArray);
      boxplot.plot();
      hold on;
      zCritical = norminv(0.975);
      plot(this.nArray, zCritical./sqrt(this.nArray), 'k--');
      plot(this.nArray, -zCritical./sqrt(this.nArray), 'k--');
      ax = fig.Children(1);
      ax.XScale = 'log';
      ax.XLabel.String = 'n';
      ax.YLabel.String = 'mean corrected z';
      ax.XLim(1) = ax.XLim(1)/10^(0.1);
      ax.XLim(2) = ax.XLim(2)*10^(0.1);
      
      %plot variance corrected z
      fig = figure;
      boxplot = Boxplots(this.varZArray, true);
      boxplot.setPosition(this.nArray);
      boxplot.plot();
      hold on;
      plot(this.nArray, chi2inv(0.975,this.nArray - 1)./(this.nArray-1), 'k--');
      plot(this.nArray, chi2inv(0.025,this.nArray - 1)./(this.nArray-1), 'k--');
      ax = fig.Children(1);
      ax.XScale = 'log';
      ax.XLabel.String = 'n';
      ax.YLabel.String = 'variance corrected z';
      ax.XLim(1) = ax.XLim(1)/10^(0.1);
      ax.XLim(2) = ax.XLim(2)*10^(0.1);
      
      %plot kurtosis corrected z
      fig = figure;
      boxplot = Boxplots(this.kurtosisZArray, true);
      boxplot.setPosition(this.nArray);
      boxplot.plot();
      hold on;
      plot(this.nArray, 3+sqrt(24)*zCritical./sqrt(this.nArray), 'k--');
      plot(this.nArray, 3-sqrt(24)*zCritical./sqrt(this.nArray), 'k--');
      ax = fig.Children(1);
      ax.XScale = 'log';
      ax.XLabel.String = 'n';
      ax.YLabel.String = 'kurtosis corrected z';
      ax.XLim(1) = ax.XLim(1)/10^(0.1);
      ax.XLim(2) = ax.XLim(2)*10^(0.1);
      
%       for i = 1:numel(this.nArray)
%         figure;
%         qqplot(this.correctedZArray{i});
%       end
      
    end
    
  end
  
  methods (Access = protected)
    
    %IMPLEMENTED: SETUP
    function setup(this)
      this.randStream = RandStream('mt19937ar','Seed', uint32(2288468478));
      this.nullMeanArray = zeros(this.nRepeat, numel(this.nArray));
      this.nullStdArray = zeros(this.nRepeat, numel(this.nArray));
      this.meanZArray = zeros(this.nRepeat, numel(this.nArray));
      this.varZArray = zeros(this.nRepeat, numel(this.nArray));
      this.kurtosisZArray = zeros(this.nRepeat, numel(this.nArray));
      this.correctedZArray = cell(1, numel(this.nArray));
    end
    
    %IMPLEMENTED: DO EXPERIMENT
    function doExperiment(this)
      
      %for each n
      for iN = 1:numel(this.nArray)
        %get n, the number of N(0,1) samples to simulate
        n = this.nArray(iN);
        
        %nRepeat times
        for iRepeat = 1:this.nRepeat
          
          z = this.randStream.randn(n,1); %simulate N(0,1);
          %instantiate empirical null, get the empirical null
          empiricalNull = EmpiricalNull(z, 0, ...
              this.randStream.randi([intmin('int32'),intmax('int32')],'int32'));
          empiricalNull.estimateNull();
          this.nullMeanArray(iRepeat, iN) = empiricalNull.getNullMean();
          this.nullStdArray(iRepeat, iN) = empiricalNull.getNullStd();
          
          %correct the z statistics, if this is the 1st repeat, save the samples
          z = (z - empiricalNull.getNullMean()) / empiricalNull.getNullStd();
          if (iRepeat == 1)
            this.correctedZArray{iN} = z;
          end
          %save the mean, variance and kurtosis of the corrected z statistics
          this.meanZArray(iRepeat, iN) = mean(z);
          this.varZArray(iRepeat, iN) = var(z);
          this.kurtosisZArray(iRepeat, iN) = kurtosis(z);
          
          this.printProgress( ((iN-1)*this.nRepeat + iRepeat) / (numel(this.nArray)*this.nRepeat) );
          
        end
        
      end

    end
    
  end
  
end

