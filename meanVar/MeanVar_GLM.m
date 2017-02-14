classdef MeanVar_GLM < VarianceModel
    %MEANVAR_GLM Abstract super class for modelling variance as gamma with known shape parameter given
    %mean greyvalue
    
    %MEMBER VARIABLES
    properties
        %shape parameter of the gamma distribution
        shape_parameter;
        y_scale;
        x_scale;
        x_shift;
        polynomial_order;
        n_step; %n_step: number of IRLS steps
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        %PARAMETERS:
            %shape_parameter: shape parameter of the gamma distribution
            %polynomial_order: polynomial order feature
        function this = MeanVar_GLM(shape_parameter,polynomial_order)
            %assign member variables
            this.shape_parameter = shape_parameter;
            this.polynomial_order = polynomial_order;
            this.n_step = 100;
        end
        
        %TRAIN CLASSIFIER
        %PARAMETERS:
            %var_train: column vector of greyvalue variance
            %mean_train: column vector of greyvalue mean
        function train(this,mean_train,var_train)

            %scale variables and get design matrix
            this.y_scale = std(var_train);
            y = var_train./this.y_scale;
            X = this.getDesignMatrix(mean_train);
            this.x_shift = mean(X(:,2));
            this.x_scale = std(X(:,2));
            X = this.getNormalisedDesignMatrix(mean_train);
            
            %IRLS SECTION
            
            %set inital parameter
            this.parameter = [-1;0];
            %assign training set size
            this.n_train = numel(var_train);
            
            %initalise variables
            eta = X*this.parameter; %systematic component
            mu = this.getMean(eta); %mean vector
            v = mu.^2 / this.shape_parameter; %variance vector
            w = 1./(v.*this.getLinkDiff(mu)); %weights in IRLS
            
            %for n_step times
            for i_step = 1:this.n_step
                
                %work out the z vector
                z = eta + (y-mu).*this.getLinkDiff(mu);
                
                %Xt_w is X'*W (W is nxn and diagonal, it is not necessary to represent the large diagonal matrix W) 
                Xt_w = X';
                for i_n = 1:this.n_train
                    Xt_w(:,i_n) = Xt_w(:,i_n).*w(i_n);
                end
                
                %update the parameter
                this.parameter = (Xt_w*X)\(Xt_w*z);
                
                %update variables
                eta = X*this.parameter; %systematic component
                mu = this.getMean(eta); %mean vector
                v = mu.^2 / this.shape_parameter; %variance vector
                w = 1./(v.*this.getLinkDiff(mu)); %weights in IRLS
                
            end
            
        end
        
        %PREDICT VARIANCE GIVEN MEAN
        %PARAMETERS:
            %x: column vector of mean greyvalue
        %RETURN:
            %variance_prediction: predicted greyvalue variance (column vector)
            %up_error: 84% percentile
            %down_error: 16% percentile
        function [variance_prediction, up_error, down_error] = predict(this,x)
            %get design matrix
            X = this.getNormalisedDesignMatrix(x);
            %work out variables
            eta = X*this.parameter;
            %work out mean, to be used for the variance prediction
            variance_prediction = this.getMean(eta) * this.y_scale;
            %get rate parameter
            gamma_scale = variance_prediction./this.shape_parameter;
            
            %work out the [16%, 84%] percentile, to be used for error bars
            up_error = gaminv(normcdf(1),this.shape_parameter,gamma_scale);
            down_error = gaminv(normcdf(-1),this.shape_parameter,gamma_scale);
            
        end
        
        %SIMULATE
        %PARAMETERS:
            %x: column vector of greyvalue means
            %parameter: parameter for GLM
        %RETURN:
            %y: column vector of simulated greyvalue variance
        function y = simulate(this,x,parameter)
            
            %get design matrix
            X = this.getDesignMatrix(x);
            
            %work out systematic component
            eta = X*parameter;
            %get rate parameter
            gamma_scale = this.getMean(eta)./this.shape_parameter;

            %simulate gamma from the natural parameter
            y = gamrnd(this.shape_parameter,gamma_scale);
        end
        
        %GET DESIGN MATRIX
        %PARAMETERS:
            %grey_values: column vector of greyvalues
        %RETURN:
            %X: n x 2 design matrix
        function X = getDesignMatrix(this,grey_values)
            X = [ones(numel(grey_values),1),grey_values.^this.polynomial_order];
        end
        
        %GET NORMALISED DESIGN MATRIX
        %Normalise the design matrix so that the 2nd column has mean zero
        %and std 1
        %PARAMETERS:
            %grey_values: column vector of greyvalues
        function X = getNormalisedDesignMatrix(this,grey_values)
            X = this.getDesignMatrix(grey_values);
            X(:,2) = (X(:,2)-this.x_shift) / this.x_scale;
        end
     
    end
    
    %ABSTRACT METHODS
    methods(Abstract)
        
        %GET LINK FUNCTION DIFFERENTATED
        %PARAMETERS:
            %mu: column vector of means
        %RETURN:
            %g_dash: colum vector of g'(mu)
        g_dash = getLinkDiff(this,mu);
        
        %GET MEAN (LINK FUNCTION)
        %PARAMETERS:
            %eta: vector of systematic components
        %RETURN:
            %mu: vector of mean responses
        mu = getMean(this,eta);
        
    end
    
end

