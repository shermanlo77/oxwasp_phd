classdef VarMeanCv < Experiment
  
  properties (SetAccess = protected)
    
    modelArray = {'linear','inverse','canonical','spline'}; %list of models
    scan; %contains the projections to work on, apply shading correction if needed
    nRepeat = 100; %number of times to repeat the experiment
    
    %array of training and test deviance
      %dim 1: for each repeat
      %dim 2: for each model
    devianceTrainingArray;
    devianceTestArray;
    
    rng; %random number generator
    
  end
  
  methods (Access = public)
    
    function this = VarMeanCv()
      this@Experiment();
    end
    
    function printResults(this)
      
    end
    
  end
  
  methods (Access = protected)
    
    %IMPLEMENTED: SETUP
    function setup(this, scan, seed)
      this.scan = scan;
      this.rng = RandStream('mt19937ar','Seed', seed);
      this.devianceTrainingArray = zeros(this.nRepeat, numel(this.modelArray));
      this.devianceTestArray = zeros(this.nRepeat, numel(this.modelArray));
    end
    
    %IMPLEMENTED: DO EXPERIMENT
    function doExperiment(this)
      
      %load the greyvalues
      greyValueArray = getGreyValue(this.scan);
      nSample = this.scan.nSample;
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
          
          %normalise the data using the training set
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
          this.devianceTrainingArray(iRepeat, iModel) = this.getDeviance(yTraining, yHat);
          
          %predict the test set, get deviance
          yHat = predict(model, X(testIndex, :));
          yTest = y(testIndex);
          this.devianceTestArray(iRepeat, iModel) = this.getDeviance(yTest, yHat);
          
          %print progress
          this.printProgress(((iModel-1)*this.nRepeat + iRepeat) ...
              /(this.nRepeat*numel(this.modelArray)));
          
        end

      end
      
    end
    
    %METHOD: GET DEVIANCE
    function deviance = getDeviance(this, y, yHat)
      deviance = 2*sum((y-yHat)./yHat - log(y./yHat));
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
        case 4
          model = fit(X(:,2),y,'smoothingspline');
      end
    end
    
    %METHOD: PREDICT
    %PARAMETERS:
      %X: design matrix, [x^{-1}, x] features
    function yHat = predict(this, model, X)
      if (isa(model, 'GeneralizedLinearModel'))
        yHat = model.predict(X);
      elseif (isa(model, 'cfit'))
        yHat = feval(model, X(:,2));
      else
        error('Model is of unknown class');
      end
    end
    
    
  end
  
end

