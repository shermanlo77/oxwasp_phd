%MIT License
%Copyright (c) 2019 Sherman Lo

%CLASS: VARIANCE MEAN CROSS VALIDATION
%Experiment to do cross validation to assess the different GLM
%
%Random images (that is selecting which replicate projections to use with replacement) were selected
    %to get var-mean data where each pixel is a data point. The data points were randomly spilt into
    %a training and test set. The GLM is trainined on the training set, and then the prediction is
    %assessed on the test set. The measure of performance used is the deviance. The experiment was
    %repeated by choosing different random images.
%Models considered here:
  %y = a + bx using identity link
  %y = a + b/x using identity link
  %y = 1/(a+b/x) using reciprocal link
%Different shading corrections were investigated: null, bw, linear
classdef VarMeanCv < Experiment
  
  properties (SetAccess = protected)
    
    %list of models
    %the use of unicode so that it renders properly in .eps output
    %see https://uk.mathworks.com/matlabcentral/answers/159732-2014b-axis-label-errors-when-printing-to-postscript
    %see https://uk.mathworks.com/matlabcentral/answers/290136-unwanted-whitespace-gap-in-figure-text-before-symbols-and-subscripts-in-matlab-2015a-for-pdf-and-e
    modelArray = {'y=a+bx','y=a+b/x','y=(a+b/x)⁻¹'};
    scanArray; %array of projections with different shading corrections to work on
    nRepeat = 100; %number of times to repeat the experiment
    
    nShadingCorrection = 3;
    
    %array of training and test deviance
      %dim 1: for each repeat
      %dim 2: for each model
      %dim 3: for each shading correction
    devianceTrainingArray;
    devianceTestArray;
    
    rng; %random number generator
    
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    function this = VarMeanCv()
      this@Experiment();
    end
    
    %IMPLEMENTED: PRINT RESULTS
    %Plots the training and test deviance for the different shading corrections and models
    function printResults(this)
      %get list of shading correction
      shadingCorrectionNameArray = cell(this.nShadingCorrection,1);
      for iShadingCorrection = 1:this.nShadingCorrection
        shadingCorrectionNameArray{iShadingCorrection} = ...
            this.scanArray{iShadingCorrection}.getShadingCorrectionStatus();
      end

      %plot the training and test deviance
      for iStat = 1:2
        isTrainingDeviance = iStat == 1;
        if (isTrainingDeviance)
          statName = 'training mean scaled deviance';
        else
          statName = 'test mean scaled deviance';
        end

        %plot the test deviance for each shading correction
        fig = LatexFigure.sub();
        legendArray = line(0,0);
        for iShadingCorrection = 1:this.nShadingCorrection
          boxplots = this.getBoxplot(-0.1 + ((iShadingCorrection-1)*0.1), isTrainingDeviance, ...
              iShadingCorrection);
          legendArray(iShadingCorrection) = boxplots.getLegendAx();
        end

        %retick the x axis
        ax = fig.Children;
        ax.XTick = 1:numel(this.modelArray);
        %label each glm with its name
        ax.XTickLabel = this.modelArray;
        ax.XLim = [0.5,numel(this.modelArray)+0.5];
        %label the axis and legend
        ylabel(statName);
        legend(legendArray,shadingCorrectionNameArray,'Location','best');

        %save the figure
        fileName = fullfile('reports','figures','varmean', ...
            strcat(class(this),'_',statName,'.eps'));
        fileName(fileName == ' ') = [];
        saveas(fig,fileName,'epsc');
      end
    end
    
  end
  
  methods (Access = protected)
    
    %IMPLEMENTED: SETUP
    function setup(this, scan, seed)
      this.scanArray = cell(3,1);
      for iShadingCorrection = 1:this.nShadingCorrection
        this.scanArray{iShadingCorrection} = feval(scan);
        switch iShadingCorrection
          case 1
            this.scanArray{iShadingCorrection}.setShadingCorrectionOff();
          case 2
            this.scanArray{iShadingCorrection}.addShadingCorrectorBw();
          case 3
            this.scanArray{iShadingCorrection}.addShadingCorrectorLinear();
        end
      end
      this.rng = RandStream('mt19937ar','Seed', seed);
      this.devianceTrainingArray = ...
          zeros(this.nRepeat, numel(this.modelArray), this.nShadingCorrection);
      this.devianceTestArray = ...
          zeros(this.nRepeat, numel(this.modelArray), this.nShadingCorrection);
      
    end
    
    %IMPLEMENTED: DO EXPERIMENT
    function doExperiment(this)
      
      %set the progress bar
      this.setNIteration(this.nRepeat*numel(this.modelArray)*this.nShadingCorrection);
      
      %for each shading correction, apply shading correction
      for iShadingCorrection = 1:this.nShadingCorrection

        %load the greyvalues
        greyValueArray = getGreyValue(this.scanArray{iShadingCorrection});
        nSample = this.scanArray{iShadingCorrection}.nSample;
        nPixel = numel(greyValueArray(:,1));
        nTrain = round(nPixel/2);

        %for each model
        for iModel = 1:numel(this.modelArray)

          %for nRepeat times
          for iRepeat = 1:this.nRepeat

            %get the greyvalues, bootstrap the images used
            imageIndex = this.rng.randi([1,nSample],nSample,1);
            X = mean(greyValueArray(:,imageIndex),2);
            %use x^-1 and x features, they will be multiplied to get higher order features
            X = X.^([-1,1]);
            y = var(greyValueArray(:,imageIndex),[],2); %get the variance of the greyvalues

            %assign random permutation for the training and test set
            randPermutation = this.rng.randperm(nPixel);
            trainingIndex = randPermutation(1:nTrain);
            testIndex = randPermutation((nTrain+1):end);

            %normalise the data
            xCentre = mean(X(trainingIndex,:),1);
            xScale = std(X(trainingIndex,:),[],1);
            X = (X-xCentre)./xScale; %noramlise
            yStd = std(y(trainingIndex));
            y = y/yStd; %noramlise

            %get the training set, train the model using it, get deviance
            XTraining = X(trainingIndex, :);
            yTraining = y(trainingIndex);
            model = this.trainModel(iModel, XTraining, yTraining);
            yHat = this.predict(model, XTraining);
            this.devianceTrainingArray(iRepeat, iModel, iShadingCorrection) = ...
                this.getDeviance(yTraining, yHat);

            %predict the test set, get deviance
            yHat = this.predict(model, X(testIndex, :));
            yTest = y(testIndex);
            this.devianceTestArray(iRepeat, iModel, iShadingCorrection) = ...
                this.getDeviance(yTest, yHat);

            %print progress
            this.madeProgress();

          end
          
        end

      end
      
    end
    
    %METHOD: GET DEVIANCE
    function deviance = getDeviance(this, y, yHat)
      deviance = (2*sum((y-yHat)./yHat - log(y./yHat))) / numel(y);
    end
    
    %METHOD: TRAIN MODEL
    %PARAMETERS:
      %X: design matrix, [x^{-1}, x] features
      %y: response vector
    function model = trainModel(this, index, X, y)
      switch index
        case 1
          model = fitglm(X,y,[0,0,0;0,1,0],'Distribution','gamma','Link','identity');
        case 2
          model = fitglm(X,y,[0,0,0;1,0,0],'Distribution','gamma','Link','identity');
        case 3
          model = fitglm(X,y,[0,0,0;1,0,0],'Distribution','gamma','Link','reciprocal');
      end
    end
    
    %METHOD: PREDICT
    %PARAMETERS:
      %X: design matrix, [x^{-1}, x] features
    function yHat = predict(this, model, X)
      if (isa(model, 'GeneralizedLinearModel'))
        yHat = model.predict(X);
      elseif (isa(model, 'KernelRegressionLookup'))
        yHat = model.predict(X(:,2));
      else
        error('Model is of unknown class');
      end
    end
    
    %METHOD: GET BOXPLOT
    %Used by printResults
    %Return and plot box plot for the training or test deviance
    %PARAMETERS:
      %xOffset: shift the box plot x
      %isTraining: true if want training error, else test error
      %iShadingCorrection: which shading correction to plot
    function boxplots = getBoxplot(this, xOffset, isTraining, iShadingCorrection)
      if (isTraining)
        boxplots = Boxplots(this.devianceTrainingArray(:,:,iShadingCorrection));
      else
        boxplots = Boxplots(this.devianceTestArray(:,:,iShadingCorrection));
      end
      boxplots.setPosition((1:numel(this.modelArray))+xOffset);
      boxplots.plot();
    end
    
  end
  
end

