%MIT License
%Copyright (c) 2019 Sherman Lo

%CLASS: COMPOUND POISSON EM ALGORITHM
%NOTE: cannot be run in parallel as all threads depend on the same rng
%Fit the compound Poisson onto simulated data using the EM algorithm
%Plots, at each step of EM, the:
  %log likelihood (marginal, not joint)
  %lambda estimate
  %alpha estimate
  %beta estimate
classdef CpEmAlgorithm < Experiment

  properties (SetAccess = private)

    seed = uint32(2225638568); %random seed

    %array of parameters to be investigated
      %dim 1: for each set
      %dim 2: lambda, alpha, beta
    parameterArray = [
      1,1,1;
      10,1,1;
      1,100,1;
      100,100,1
      ];
    nParameter; %number of parameters in parameterArray, numel(parameterArray(:,1))
    %array to plot zero count for each of the parameters
    isPlotZeroArray = [true, false, true, false];

    nSimulation = 1000; %simulation sample size
    nRepeat = 10; %number of times to repeat EM
    nStep = 10; %number of EM steps

    %declare array of lnL, lambda, alpha and beta for each step of EM and each repeat
      %dim 1: for each step of EM + 1
      %dim 2: for each repeat of the experiment
      %dim 3: for each parameter set
    lnLArray;
    lambdaArray;
    alphaArray;
    betaArray;
    %array of std for each of the estimators
      %dim 1: lambda, alpha, beta
      %dim 2: for each parameter set
    stdArray;

  end

  methods (Access = public)

    %IMPLEMENTED: PRINT RESULTS
    function printResults(this)

      figureLocation = fullfile('reports','figures','compoundpoisson');

      for iParameter = 1:this.nParameter

        %get the parameter
        lambda = this.parameterArray(iParameter,1);
        alpha = this.parameterArray(iParameter,2);
        beta = this.parameterArray(iParameter,3);

        %instatise a compound poisson for the purposes of naming the figure file name
        compoundPoissonTrue = CompoundPoisson();
        compoundPoissonTrue.setParameters(lambda, alpha, beta);

        %plot the log likelihood
        fig = LatexFigure.subLoose();
        ax = gca;
        plot(0:this.nStep, -this.lnLArray(:,:,iParameter), 'b');
        xlabel('number of EM steps');
        ylabel('-lnL');
        xlim([0,this.nStep]);
        ax.YAxis.Exponent = floor(log10(abs(ax.YLim(2))));
        print(fig, ...
            fullfile(figureLocation, strcat(class(this),'_',compoundPoissonTrue.toString(),...
            '_lnL.eps')),'-depsc','-loose');

        %plot lambda
        fig = LatexFigure.subLoose();
        plot(0:this.nStep, this.lambdaArray(:,:,iParameter), 'b');
        xlabel('number of EM steps');
        ylabel('\lambda');
        xlim([0,this.nStep]);
        print(fig, ...
            fullfile(figureLocation, strcat(class(this),'_',compoundPoissonTrue.toString(),...
            '_lambda.eps')),'-depsc','-loose');

        %plot alpha
        fig = LatexFigure.subLoose();
        plot(0:this.nStep, this.alphaArray(:,:,iParameter), 'b');
        xlabel('number of EM steps');
        ylabel('\alpha');
        xlim([0,this.nStep]);
        print(fig, ...
            fullfile(figureLocation, strcat(class(this),'_',compoundPoissonTrue.toString(),...
            '_alpha.eps')),'-depsc','-loose');

        %plot beta
        fig = LatexFigure.subLoose();
        plot(0:this.nStep, this.betaArray(:,:,iParameter), 'b');
        xlabel('number of EM steps');
        ylabel('\beta');
        xlim([0,this.nStep]);
        print(fig, ...
            fullfile(figureLocation, strcat(class(this),'_',compoundPoissonTrue.toString(),...
            '_beta.eps')),'-depsc','-loose');

      end
    end

  end

  methods (Access = protected)

    %IMPLEMENTED: SETUP
    function setup(this)

      %get number of parameter set
      this.nParameter = numel(this.parameterArray(:,1));

      %declare array of lnL, lambda, alpha and beta for each step of EM and each repeat
        %dim 1: for each step of EM
        %dim 2: for each repeat of the experiment
        %dim 3: for each parameter set
      this.lnLArray = zeros(this.nStep+1, this.nRepeat, this.nParameter);
      this.lambdaArray = zeros(this.nStep+1, this.nRepeat, this.nParameter);
      this.alphaArray = zeros(this.nStep+1, this.nRepeat, this.nParameter);
      this.betaArray = zeros(this.nStep+1, this.nRepeat, this.nParameter);
      %array of std for each of the estimators
        %dim 1: lambda, alpha, beta
        %dim 2: for each parameter set
      this.stdArray = zeros(3, this.nParameter);
    end

    %IMPLEMENTED: DO EXPERIMENT
    function doExperiment(this)

      %set random number generator
      %NOTE: cannot be run in parallel as all threads depend on the same rng
      rng(this.seed,'twister');

      %for each parameter
      for iParameter = 1:this.nParameter

        %get the parameter
        lambda = this.parameterArray(iParameter,1);
        alpha = this.parameterArray(iParameter,2);
        beta = this.parameterArray(iParameter,3);

        %instantise a compound Poisson with the true parameters
        compoundPoissonTrue = CompoundPoisson();
        compoundPoissonTrue.setN(this.nSimulation);
        compoundPoissonTrue.setParameters(lambda, alpha, beta);

        %get the standard error of the estimators of the 3 parameters using the Fisher's information
            %matrix
        this.stdArray(:,iParameter) = sqrt(diag(inv(compoundPoissonTrue.getFisherInformation())))';

        %for n_repeat times
        for iRepeat = 1:this.nRepeat

          %simulate n compound poisson varibales
          X = CompoundPoisson.simulate(this.nSimulation,lambda,alpha,beta);

          %set up a compound poisson random variable
          compoundPoisson = CompoundPoisson();
          compoundPoisson.setParameters(lambda,alpha,beta);
          compoundPoisson.addData(X);
          compoundPoisson.initaliseEM();

          %save the log likelihood, lambda, alpha and beta before EM
          this.lnLArray(1, iRepeat, iParameter) = compoundPoisson.getMarginallnL();
          this.lambdaArray(1, iRepeat, iParameter) = compoundPoisson.lambda;
          this.alphaArray(1, iRepeat, iParameter) = compoundPoisson.alpha;
          this.betaArray(1, iRepeat, iParameter) = compoundPoisson.beta;

          %for n_step times
          for iStep = 1:this.nStep
            %take a E and M step
            compoundPoisson.EStep();
            compoundPoisson.MStep();
            %save the log likelihood, lambda, alpha and beta before EM
            this.lnLArray(iStep+1, iRepeat, iParameter) = compoundPoisson.getMarginallnL();
            this.lambdaArray(iStep+1, iRepeat, iParameter) = compoundPoisson.lambda;
            this.alphaArray(iStep+1, iRepeat, iParameter) = compoundPoisson.alpha;
            this.betaArray(iStep+1, iRepeat, iParameter) = compoundPoisson.beta;
          end
        end

        this.printProgress(iParameter / this.nParameter);

      end

    end

  end

end
