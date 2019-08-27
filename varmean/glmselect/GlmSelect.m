%MIT License
%Copyright (c) 2019 Sherman Lo

%ABSTRACT CLASS: GLM SELECT EXPERIMENT
%Does forward stepwise regression to select a GLM model for the var-mean data
%
%The following was chosen:
  %gamma distributed GLM
  %link functions: identity, reciprocal
%
%The var-mean data was obtained by selecting random replicated projections with replacement. A GLM
    %is fitted onto the var-mean data with just a constant. Then an extra term is added, either a
    %higher positive polynomial term or a higher negative polynomial term. If the extra term reduce
    %the criterion, then add that term. This is continued until the criterion doesn't decrease. The 
    %experiment is repeated by calculating the var-mean data again using a different random
    %selection of replicated projections. Various shading corrections were investigated: null, bw,
    %linear
classdef GlmSelect < Experiment
  
  properties (SetAccess = protected)
    
    scanArray; %scan object with shading correction initalised
    nShadingCorrection = 3; %number of shading corrections
    
    %contains lowest and highest polynomial order selected
      %dim 1: [negative, positive] polynomial orders
      %dim 2: for each repeat
      %dim 3: for each link function
      %dim 4: for each shading correction
    selectedPolynomial;
    
    %contains the criterion
      %dim 1: for each repeat
      %dim 2: for each link function
      %dim 3: for each shading correction
    criterionArray;
    
    linkArray = {'identity', 'reciprocal'}; %array of link functions
    nRepeat = 10; %number of repeats
    rng; %random number generator
    
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    function this = GlmSelect()
      this@Experiment();
    end
    
    %IMPLEMENTED: PRINT RESULTS
    function printResults(this)
      
      %for each shading correction
      for iShadingCorrection = 1:this.nShadingCorrection
        scan = this.scanArray{iShadingCorrection};
        
        %for each link function
        for iLink = 1:numel(this.linkArray)
          
          %declare empty array to store polynomial properties
          %store string identifier for the selected polynomial eg '-1 to 0'
          polynomialKey = cell(0,0);
          %store number of votes this polynomial has, entries correspond to polynomialKey
          polynomialCount = [];
          
          %for each repeat
          for iRepeat = 1:this.nRepeat
            
            if (this.selectedPolynomial(1,iRepeat,iLink,iShadingCorrection) == 0)
              minusSign = '';
            else
              minusSign = '-';
            end
            
            %make a sting identifier for this selected polynomial
            selectedPolyString = cell2mat({ ...
              minusSign, num2str(this.selectedPolynomial(1,iRepeat,iLink,iShadingCorrection)), ...
              ' to ', ...
              num2str(this.selectedPolynomial(2,iRepeat,iLink,iShadingCorrection)) ...
            });
            
            %find if this selected polynomial has already been stored
            isGotPolynomial = false;
            for iKey = 1:numel(polynomialKey)
              %if this selected polynomial has already been storied, then increment the polynomial
                  %count
              if (strcmp(polynomialKey{iKey},selectedPolyString))
                isGotPolynomial = true;
                polynomialCount(iKey) = polynomialCount(iKey)+1;
                break; 
              end
            end
            %if this selected polynomial hasn't been stored, append it to array of polynomials
            if (~isGotPolynomial)
              polynomialKey{numel(polynomialKey)+1} = selectedPolyString;
              polynomialCount(numel(polynomialKey)) = 1; %set counter to 1
            end

          end

          %quote deviance
          devianceQuote = siUncertainity(mean(this.criterionArray(:,iLink,iShadingCorrection)), ...
              std(this.criterionArray(:,iLink,iShadingCorrection)), 2);
          %quote polynomial with the most votes
          [maxCount,maxKey] = max(polynomialCount);
          
          prefix = strcat(class(this), scan.getShadingCorrectionStatus(),'_',this.linkArray{iLink});
          
          file = fopen(fullfile('reports','figures','varmean',strcat(prefix, 'order.txt')),'wt');
          fprintf(file, polynomialKey{maxKey});
          fclose(file);
          
          file = fopen(fullfile('reports','figures','varmean',strcat(prefix, 'vote.txt')),'wt');
          fprintf(file, num2str(maxCount));
          fclose(file);
          
          file = fopen(fullfile('reports','figures','varmean',strcat(prefix, ...
              'criterion.txt')),'wt');
          fprintf(file, devianceQuote);
          fclose(file);
          
        end
      
      end
      
    end
    
  end
  
  methods (Access = protected)
    
    %IMPLEMENTED: SETUP
    function setup(this, scan, seed)
      this.scanArray = cell(this.nShadingCorrection, 1);
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
      this.selectedPolynomial = zeros(2, this.nRepeat, numel(this.linkArray), ...
          this.nShadingCorrection);
      this.criterionArray = zeros(this.nRepeat, numel(this.linkArray), this.nShadingCorrection);
    end
    
    %IMPLEMENTED: DO EXPERIMENT
    function doExperiment(this)
      
      %set up progress bar
      this.setNIteration(this.nShadingCorrection * numel(this.linkArray) * this.nRepeat);
      
      %for each shading correction
      for iShadingCorrection = 1:this.nShadingCorrection
        
        %load the greyvalues with the corresponding shading correction
        scan = this.scanArray{iShadingCorrection};
        greyValueArray = getGreyValue(scan);
        n = scan.nSample;

        %for each link function
        for iLink = 1:numel(this.linkArray)
          %for nRepeat times
          for iRepeat = 1:this.nRepeat
            
            %get the greyvalues, bootstrap the images used
            imageIndex = this.rng.randi([1,n],n,1);
            X = mean(greyValueArray(:,imageIndex),2);
            %use x^-1 and x features, they will be multiplied to get higher order features
            X = X.^([-1,1]);
            X = (X-mean(X,1))./std(X,[],1); %noramlise
            y = var(greyValueArray(:,imageIndex),[],2); %get the variance of the greyvalues
            y = y/std(y); %noramlise
            
            %boolean to indicate if a model has been selected
            isSelect = false;
            
            %fit constant and get the initial criterion
            glm = fitglm(X,y,'constant','Distribution','gamma','Link',this.linkArray{iLink});
            criterion = this.getCriterion(glm);
            
            %contains the lowest and highest polynomial order
              %element 1: negative order
              %element 2: positive order
            %the two elements will increment in the forward stepwise regression
            orders = [0; 0];
            
            %while a model hasn't been selected
            while (~isSelect)
              
              %add a positive polynomial with a term higher
              glmUpper = glm.addTerms([0, orders(2)+1, 0]); %3rd element represent y
              criterionUpper = this.getCriterion(glmUpper); %get the criterion
              
              %add a negative polynomial with a term higher
              glmLower = glm.addTerms([orders(1)+1, 0, 0]); %3rd element represent y
              criterionLower = this.getCriterion(glmLower); %get the criterion
              
              %if none of the new crierion is lower, model selected
              if ( (criterion<criterionUpper) && (criterion<criterionLower) )
                isSelect = true;
              %else choose the lowest criterion and increment the corresponding polynomial order
              else
                if (criterionUpper < criterionLower)
                  orders(2) = orders(2) + 1;
                  criterion = criterionUpper;
                else
                  orders(1) = orders(1) + 1;
                  criterion = criterionLower;
                end
              end
            end
            
            %save the selected polynomial and the criterion
            this.selectedPolynomial(:, iRepeat, iLink, iShadingCorrection) = orders;
            this.criterionArray(iRepeat, iLink, iShadingCorrection) = criterion;
            
            %print progress bar
            this.madeProgress();
            
          end
          
        end
      
      end
      
    end
    
    %METHOD: LOG LIKELIHOOD
    %Evaluate the log likelihood with KNOWN shape parameter
    function lnL = getLogLikelihood(this, glm)
      alpha = (this.scanArray{1}.nSample - 1) / 2;
      yHat = table2array(glm.Fitted(:,1));
      y = table2array(glm.Variables(:,end));
      lnL = sum((alpha*log(alpha) - gammaln(alpha)) - alpha*log(yHat)  + (alpha-1)*log(y) ...
          - alpha*(y./yHat));
    end
    
  end
  
  methods (Abstract, Access = protected)
    
    %ABSTRACT METHOD: GET CRITERION
    %Return the criterion, eg AIC and BIC
    aic = getCriterion(this, glm)
    
  end
  
end

