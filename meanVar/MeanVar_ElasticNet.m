classdef MeanVar_ElasticNet < MeanVar_GLM

    properties
        alpha;
        lambda;
    end
    
    methods
        
        %CONSTRUCTOR
        function this = MeanVar_ElasticNet(shape_parameter,polynomial_order,link_function,alpha, lambda)
            this@MeanVar_GLM(shape_parameter,polynomial_order,link_function);
            this.alpha = alpha;
            this.lambda = lambda;
        end
        
        function updateParameter(this, w, X, z, y)
            %z = X*this.parameter;
            this.parameter(1) = sum((z - X(:,2:end)*this.parameter(2:end)).*w)/sum(w);
            
            for i_order = 2:(this.n_order+1)
                                    
                partial_index = 1:(this.n_order+1);
                partial_index(i_order) = [];

                z_partial = X(:,partial_index)*this.parameter(partial_index);
                %[~, ~, z_partial] = this.getIRLSStatistics(X(:,partial_index), y, this.parameter(partial_index));

                lambda_i = this.lambda;

                s = this.softThreshold( (w.*X(:,i_order))' * (z - z_partial) , lambda_i * this.alpha );

                if s ~= 0
                    this.parameter(i_order) = s / ( (w'*(X(:,i_order).^2)) + lambda_i*(1-this.alpha) );
                else
                    this.parameter(i_order) = 0;
                end


                if i_order ~= (this.n_order+1)
                    [~, w, z] = this.getIRLSStatistics(X, y, this.parameter);
                end
            end
                

        end
        
        
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

