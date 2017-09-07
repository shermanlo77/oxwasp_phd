%MEAN VARIANCE MODEL USING GLM ELASTIC NET
%Extends MeanVar_GLM
%Fits GLM using elastic net regularisation
classdef MeanVar_ElasticNet < MeanVar_GLM

    %MEMBER VARIBLES
    properties
        alpha; %elastic net parameter between 0 and 1; 0 for ridge regression, 1 for lasso
        lambda; %tuning parameter
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        %PARAMETERS
            %shape_parameter: scalar, gamma shape parameter
            %polynomial_order: column vector of polynomial orders
            %link_function: link function obkect
            %alpha: elastic net parameter
            %lambda: tuning parameter
        function this = MeanVar_ElasticNet(shape_parameter,polynomial_order,link_function,alpha, lambda)
            %call superclass
            this@MeanVar_GLM(shape_parameter,polynomial_order,link_function);
            %assign member variables
            this.alpha = alpha;
            this.lambda = lambda;
        end
        
        %OVERRIDE: UPDATE PARAMETER
        %Does coordinate descent to update the parameters
        function updateParameter(this, w, X, z, y)
            
            %update the intercept term
            this.parameter(1) = sum((z - X(:,2:end)*this.parameter(2:end)).*w)/sum(w);
            
            %for the other parameter terms
            for i_order = 2:(this.n_order+1)
                
                %get the array of integers from 1 to n_order + 1, except for this parameter term
                partial_index = 1:(this.n_order+1);
                partial_index(i_order) = [];

                %do regression without this parameter term
                z_partial = X(:,partial_index)*this.parameter(partial_index);

                %get the tuning parameter times number of training data
                lambda_i = this.n_train * this.lambda;

                %update this parameter
                s = this.softThreshold( (w.*X(:,i_order))' * (z - z_partial) , lambda_i * this.alpha );
                if s ~= 0
                    this.parameter(i_order) = s / ( (w'*(X(:,i_order).^2)) + lambda_i*(1-this.alpha) );
                else
                    this.parameter(i_order) = 0;
                end

                %after parameter update, update the weights and the z response
                %if this is the last iteration, don't do that as it will be done in the superclass method train()
                if i_order ~= (this.n_order+1)
                    [~, w, z] = this.getIRLSStatistics(X, y, this.parameter);
                end
            end
        end
        
        %SOFT THRESHOLD
        %Method of coordinate descent
        function s = softThreshold(this, z, gamma)
            if gamma >= abs(z)
                s = 0;
            elseif z>0
                s = z-gamma;
            else
                s = z+gamma;
            end
        end
        
    end
    
end

