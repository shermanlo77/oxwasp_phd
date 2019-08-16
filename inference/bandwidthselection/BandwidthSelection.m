%MIT License
%Copyright (c) 2019 Sherman Lo

%CLASS: EXPERIMENT Z NULL
%Experiment for simulating n N(0,1) and storing the emperical null mean and var
%Different n and parzen kernel width are investigated
%Plotted are log MSE vs kernel width vs log n, MSE for the null mean and null var
classdef BandwidthSelection < Experiment
  
  %MEMBER VARIABLES
  properties (SetAccess = private)
    
    rng = RandStream('mt19937ar', 'Seed', uint32(2055696007)); %random number generator
    nArray = unique(flip(round(linspace(0.0631, 0.63, 300).^(-5))))'; %array of n (sample size)
    kArray = linspace(0.09, 1.5, 30)'; %array of kernel width
    nRepeat = 100;
    
    %array to store the mode and half width estimation using the Parzen density estimation
    %dim 1: for each repeat
    %dim 2: for each kernel width
    %dim 3: for each n or each sample size
    meanArray;
    stdArray;
    
  end
  
  %METHODS
  methods (Access = public)
    
    %CONSTRUCTOR
    function this = BandwidthSelection()
      this@Experiment();
    end
    
    %IMPLEMENTED: PRINT RESULTS
    function printResults(this)
      
      %meshgrid for n and k
      [nPlot, kPlot] = meshgrid(log10(this.nArray), this.kArray);
      factor_array = [0.9, 1.06, 1.144]; %array of fudge factors
      
      %for the mode estimation, then half width estimation
      for i_array = 1:3
        
        %get the corresponding array
        if i_array == 1
          array = squeeze(median(this.meanArray.^2));
          zLabel = 'median squared error';
        elseif i_array == 2
          array = squeeze(median((this.stdArray-1).^2));
          zLabel = 'median squared error';
        else
          array = squeeze(median(this.stdArray));
          zLabel = '\sigma_0 estimate';
        end
        
        %surf plot the error vs k and log n
        fig = LatexFigure.sub();
        surf(nPlot, kPlot, array, 'EdgeColor', 'none');
        %label axis
        xlabel('bandwidth');
        ylabel('log(n)');
        zlabel(zLabel);
        hold on;
        
        %plot the path of the rule of thumb
        %logn_path is array of logn to evaluate the rule of thumb
        logn_path = interp1((1:numel(this.nArray))',log10(this.nArray), ...
            linspace(1,numel(this.nArray),1000*numel(this.nArray))');
        if i_array ~= 3
          %for each fudge factor
          for i = 1:numel(factor_array)
            %for each logn_path, work out the rule of thumb k
            k_path = factor_array(i)*((10.^logn_path).^(-1/5));
            %then for each k and logn pair, interpolate to get the value of the array
            path = interp2(nPlot,kPlot,array,logn_path,k_path);
            %plot the error along the rule of thumb
            plot3(logn_path,k_path,path,'LineWidth',2');
          end
          ax = gca;
          ax.Children(numel(factor_array)).Color = [0,1,1];
        end
        %set the axis and view angle
        xlim(log10([this.nArray(1), this.nArray(end)]));
        ylim(this.kArray([1,numel(this.kArray)]));
        
        %for the 3rd array
        if i_array==3
          hold on;
          %meshplot the true value of the null variance
          ax = surf(nPlot,kPlot,ones(size(array)), 'EdgeColor', 'none');
          ax.FaceColor = [0.75,0.75,0.75];
          %ax.FaceAlpha = 0;
          view(45,35.264);
        else
          view(135,35.264);
          %plot legend
          ax = gca;
          legend(ax.Children([3,2,1]),{'0.9','1.06','1.144'},'Location','best');
        end
        
        %save the figure;
        saveas(fig,fullfile('reports','figures','inference', ...
            strcat(this.experimentName,'_error',num2str(i_array),'.eps')),'epsc');
        
      end
    end
    
  end
  
  %PROTECTED METHODS
  methods (Access = protected)
    
    %IMPLEMENTED: SETUP
    function setup(this)
      %declare array to store the mode and half width estimation using the Parzen density estimation
        %dim 1: for each repeat
        %dim 2: for each kernel width
        %dim 3: for each n or each sample size
      this.meanArray = zeros(this.nRepeat, numel(this.kArray), numel(this.nArray) );
      this.stdArray = zeros(this.nRepeat, numel(this.kArray), numel(this.nArray) );
    end
    
    %IMPLEMENTED: DO EXPERIMENT
    function doExperiment(this)
      
      %set progress bar
      this.setNIteration(this.nRepeat * numel(this.nArray) * numel(this.kArray));
      
      %for every n in nArray
      for iN = 1:numel(this.nArray)
        
        %get n or sample size
        n = this.nArray(iN);
        
        %for every kernel width
        for iK = 1:numel(this.kArray)
          
          %get the kernel width
          k = this.kArray(iK);
          
          for iRepeat = 1:this.nRepeat
           
            %simulate n N(0,1)
            x = this.rng.randn(n,1);

            %instantise z tester
            empiricalNull = EmpiricalNull(x, 0, ...
                this.rng.randi([intmin('int32'),intmax('int32')], 'int32'));
            empiricalNull.setBandwidth(k);
            empiricalNull.estimateNull();

            %save the mode and half width estimation
            this.meanArray(iRepeat, iK, iN) = empiricalNull.getNullMean();
            this.stdArray(iRepeat, iK, iN) = empiricalNull.getNullStd();

            %progress bar
            this.madeProgress();
            
          end
          
        end
        
      end
      
    end
     
  end
  
end

