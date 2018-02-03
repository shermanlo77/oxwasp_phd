%CLASS: LINEAR REGRESSION
%1D regression
classdef LinearRegression < handle
    
    %MEMBER VARIABLES
    properties (SetAccess = protected)
        beta; %2 column vector, for the parameter
        x_shift; %central normalisation constant for x
        x_scale; %scale normalisation constant for x
        y_shift; %central normalisation constant for y
        y_scale; %scale normalisation constant for y
        w; %column vector of weights
        n_feature; %number of features
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        function this = LinearRegression()
            %set the number of features
            this.n_feature = 2;
        end
        
        %METHOD: ADD WEIGHTS
        %PARAMETERS:
            %w: column vector of weights
        function addWeights(this, w)
            %normalise the weights and assign it
            this.w = numel(w)*w/sum(w);
        end
        
        %METHOD: TRAIN
        %Train the linear regression
        %PARAMETERS:
            %x: column vector of features
            %y: column vector of responses
        function train(this, x, y)
            %get number of data
            n = numel(x);
            %if the weight vector is empty, fill it with ones
            if isempty(this.w)
                this.w = ones(n,1);
            end
            
            %scale the response variables
            this.y_scale = std(y);
            this.y_shift = mean(y);
            y = (y - this.y_shift) / this.y_scale;
            
            %get the design matrix
            X = this.getDesignMatrix(x);
            %normalise the features to have mean 0 and std 1
            this.x_shift = mean(X(:,2:end),1);
            this.x_scale = std(X(:,2:end),true,1); %normalise by n
            %get the normalised design matrix
            X = this.normaliseDesignMatrix(X);
            
            %get the number of data with positive weights
            n_sub = sum(this.w>0);
            %normalise the weights so that the sum = n
            w_sub = (n_sub/sum(this.w)) * this.w;

            %get the square root of the weights
            w_root = sqrt(w_sub);

            %get the weighted design matrix
            X = X.*repmat(w_root,1,this.n_feature);

            %get the weighted response vector
            y = w_root .* y;

            %estimate the regression parameter using QR
            this.beta = X \ y;
            
        end
        
        %METHOD: PREDICT
        %Predict given x
        %PARAMETERS:
            %x: column vector of features
        %RETURN:
            %y: column vector of the regression evaluated for each value in x
        function y = predict(this, x)
            %for the design matrix
            X = this.getNormalisedDesignMatrix(x);
            %get the regression
            y = X * this.beta;
            %un-normalise the regression
            y = (y * this.y_scale) + this.y_shift;
        end
        
    end
    
    %PROTECTED METHODS
    methods (Access = protected)
        
        %METHOD: GET DESIGN MATRIX
        %Returns a design matrix with polynomial features given a column vector of explanatory variables
        %PARAMETERS:
            %x: column vector of greyvalues
        %RETURN:
            %X: n x this.n_order + 1 design matrix
        function X = getDesignMatrix(this,x)
            %declare design matrix
            X = zeros(numel(x),this.n_feature);
            %first column is a constant
            X(:,1) = 1;
            X(:,2) = x;
        end
        
        %METHOD: GET NORMALISED DESIGN MATRIX
        %Return a normalised design matrix given a vector of data (with polynomial features)
        %PARAMETERS:
            %x: column vector of greyvalues
        %RETURN:
            %X: normalised design matrix (nxp matrix, 1st column constants)
        function X = getNormalisedDesignMatrix(this,x)
            X = this.getDesignMatrix(x);
            X = this.normaliseDesignMatrix(X);
        end
        
        %METHOD: NORMALISE DESIGN MATRIX
        %Normalise a given design matrix (1st column constants) to have
            %columns with mean 0
            %columns with var 1 (n divisor)
        %PARAMETERS
            %X: design matrix (nxp matrix, 1st column constant and ignored)
        %RETURN
            %X: normalised design matrix
        function X = normaliseDesignMatrix(this,X)
            n = numel(X(:,1));
            X(:,2:end) = ( X(:,2:end)- repmat(this.x_shift, n, 1 ) ) ./ repmat(this.x_scale, n, 1);
        end
        
    end
    
end

