%CLASS: EXPERIMENT Z NULL MSE
%Experiment for investigating the performance of the emperical null var
%ln mean squared error vs kernel width are plotted and fitted using local quadratic regression
%the optimal kernel width is found using the minimum of the fitted curve
%optimal kernel width vs n^(-1/5) is plotted and straight line is plotted
%boxplot is plotted by bootstraping the ln mean squared error vs kernel width data
%sensitivty analysis is done by varying the parameter of the local quadratic regression
%
%Plotted are:
  %ln mean squared error vs kernel width for all n
  %optimal kernel width vs n^(-1/5)
  %gradient and intercept vs smoothness of local quadratic regression
classdef BandwidthSelection2 < Experiment
  
  %MEMBER VARIABLES
  properties (SetAccess = protected)
    rng; %random number generator
    kArray; %array of values of kernel widths to investigate
    kOptimal; %optimal kernel width for each n
    
    %the fitted object, cell array
    %dim 1: for each n
    errorFit;
    
    %the glm fit for optimal bandwidth vs n^{-1/5}
    fitter;
  end
  
  %METHODS
  methods (Access = public)
    
    %CONSTRUCTOR
    function this = BandwidthSelection2()
      this@Experiment();
    end
    
    %IMPLEMENTED: RESULTS
    function printResults(this)
      
      %get the results from the previous experiment
      previous = this.getPreviousResult();
      
      colorOrder = get(groot,'defaultAxesColorOrder');
      colour = colorOrder(1,:);
      colour2 = colorOrder(2,:);
      yLim = [-15,5];
      
      nPlot = 100;
      
      %for each n
      for iN = 1:numel(previous.nArray)
        %get the ln MSE
        array = this.getObjective(previous.stdArray(:,:,iN));
        fig = LatexFigure.sub();
        %boxplot the ln MSE for each kernel width
        boxPlot = Boxplots(array);
        %set the values of the kernel width
        boxPlot.setPosition(previous.kArray);
        boxPlot.plot();
        hold on;
        %plot the local quadratic regression
        kPlot = linspace(this.kArray(1), this.kArray(end), numel(this.kArray)*nPlot);
        plot(kPlot,feval(this.errorFit{iN},kPlot));
        plot(this.kOptimal(iN)*ones(1,2), yLim, '--'); 
        %label axis and graph
        ylabel('ln squared error');
        xlabel('bandwidth');
        title(strcat('log n=',num2str(log10(previous.nArray(iN)))));
        ylim(yLim);
        saveas(fig,fullfile('reports','figures','inference', ...
            strcat(this.experimentName,'_plot',num2str(iN),'.eps')),'epsc');
      end
      
      %see random method in classreg.regr.CompactGeneralizedLinearModel
      x = (previous.nArray).^(-1/5);
      y = this.fitter.predict(x);
      shapeParameter = 1/this.fitter.Dispersion;
      gammaScale = y/shapeParameter;
            
      fig = LatexFigure.main();
      scatter(x, this.kOptimal, 'x');
      hold on;
      plot(x, y, 'Color', colour2);
      plot(x, gaminv(0.975, shapeParameter, gammaScale), 'Color', colour2, 'LineStyle',':');
      plot(x, gaminv(0.025, shapeParameter, gammaScale), 'Color', colour2, 'LineStyle',':');
      xlabel('n^{-1/5}');
      ylabel('optimal bandwidth');
      ax = gca;
      ax.YLim(1) = 0;
      saveas(fig,fullfile('reports','figures','inference', ...
          strcat(this.experimentName,'_ruleOfThumb','.eps')),'epsc');
      
      %save the intercept and gradient
      coefficients = this.fitter.Coefficients.Estimate;
      standardError = this.fitter.Coefficients.SE;
      latex_table = LatexTable(coefficients, standardError,{'Intercept','Gradient'} , {'Estimate'});
      latex_table.print(fullfile('reports','figures','inference', ...
          strcat(this.experimentName,'_bandwidthEstimate.txt')));
      
    end
    
  end
  
  %PROTECTED METHODS
  methods (Access = protected)
    
    %IMPLEMENTED: SET UP EXPERIMENT
    function setup(this)
      
      %get the results from the ZNull experiment
      previous = this.getPreviousResult();
      previous.run();
      
      %assign member variables
      this.kArray = previous.kArray;
      this.kOptimal = zeros(numel(previous.nArray), 1);
      this.errorFit = cell(numel(previous.nArray), 1);
    end
    
    %IMPLEMENTED: DO EXPERIMENT
    function doExperiment(this)
      %get the results of the previous experiment
      previous = this.getPreviousResult();
      %set progress bar
      this.setNIteration(numel(previous.nArray));

      %for each n
      for iN = 1:numel(previous.nArray)
        %get the ln mse
        lnMse = this.getObjective(previous.stdArray(:,:,iN));
        %declare array of kernel widths, taking into account the n_repeat in the previous experiment
        x = repmat(previous.kArray, previous.nRepeat, 1);
        %declare array of ln_mse
        y = reshape(lnMse',[],1);
        %remove any NAN
        x(isnan(y)) = [];
        y(isnan(y)) = [];
        
        %fit y vs x (ln mse vs kernel width) using the regression
        spline = fit(x,y,'smoothingspline');
        %get the fitted regression for each kernel width
        this.errorFit{iN} = spline;
        %find the optimal kernel width
        this.kOptimal(iN) = fminsearch(spline, 0.9 * previous.nArray(iN)^(-1/5));

        %update the progress bar
        this.madeProgress();

      end
      
      %get array of values of n^(-1/5)
      this.fitter = fitglm(previous.nArray.^(-1/5), this.kOptimal, ...
          'Distribution', 'gamma', 'Link', 'identity');  
      
    end
    
    %METHOD: GET PREVIOUS
    %RETURN:
    %previous: Experiment object from Z Null experiment
    function previous = getPreviousResult(this)
      previous = BandwidthSelection();
    end
    
    %METHOD: GET OBJECTIVE
    %Return the ln mse
    %PARAMETERS:
    %var_array: array of estimated variances
    %RETURN:
    %objective: array of ln mse of the estimate variances
    function objective = getObjective(this, stdArray)
      objective = log((stdArray-1).^2);
    end
    
    
  end
  
end

