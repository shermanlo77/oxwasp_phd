classdef CompoundPoisson_normal < CompoundPoisson
    
    methods
        
        %CONSTRUCTOR
        function this = CompoundPoisson_normal(t)
            this = this@CompoundPoisson(t);
        end
 
        %GET NORMAL APPROXIMATION DENSITY
        %For a range of the domain and given parameters, return the
        %Normal density.
        %PARAMETERS:
            %x_min: lowest part of the domain
            %x_max: highest part of the domain
            %n_point: number of points between x_min and x_max
            %nu: poisson parameter
            %alpha: gamma shape parameter
            %lambda: gamma rate parameter
        %RETURN:
            %f: row vector of size n_points containing the saddle density for each point
            %x: row vector of size n_points, linspace(x_min,x_max,n_point)
        function [f,x] = getDensity(this,x_min,x_max,n_point,nu,alpha,lambda)
            
            x = linspace(x_min,x_max,n_point); %get equally spaced points in the domain
            
            %work out the pdf for each point in x
            f = normpdf(x,alpha*nu*this.t/lambda,sqrt(nu*this.t*alpha*(alpha+1))/lambda);

        end
        
        %MINUS LOG LIKELIHOOD
        %PARAMETERS:
            %parameters: 3 vector containing the parameters
            %X: row vector containing data
        %RETURN
            %log_likelihood: -lnL
            %grad: gradient
        function [log_likelihood, grad] = lnL(this,parameters,X)
            
            %get the parameters
            nu = parameters(1);
            alpha = parameters(2);
            lambda = parameters(3);

            %get the number of data
            n = numel(X);
            
            %work out the minus log likelihood
            term_1 = n*log(lambda);
            term_2 = -n*sum(log([nu*this.t,alpha,alpha+1]))/2;
            term_3 = -sum((X-alpha*nu*this.t/lambda).^2)*lambda^2/(2*nu*this.t*alpha*(alpha+1));
            
            log_likelihood = -sum([term_1,term_2,term_3]);
            
            %work out the gradient if requested
            %if require the gradient, return it
            if nargout > 1
                %initalise a 3 x n matrix
                %each column is the gradient for each data point
                grad = zeros(3,n);
                %for each datapoint, work out the gradient
                for i = 1:n
                    grad(:,i) = this.gradient(parameters,X(i));
                end
                %sum the gradient over each data point
                grad = sum(grad,2);
            end
            
        end
        
        %GRADIENT
        %Evaluate the gradient of -lnL for a given single datapoint
        %PARAMETERS:
            %parameter: 3 row vector containing the parameters
            %x: datapoint
        function grad = gradient(this,parameter,x)
            %if all the parameters are positive
            if all(parameter)>0
                %extract the parameters
                nu = parameter(1);
                alpha = parameter(2);
                lambda = parameter(3);
                %declare a 3 vector
                grad = zeros(1,3);
                %work out the gradient using the nested functions
                grad(1) = - sum([-alpha*this.t*nu,-alpha^2*nu*this.t*(1+nu*this.t),x^2*lambda^2]) / (2*alpha*(alpha+1)*nu^2*this.t);
                grad(2) = - sum([-2*alpha^3*nu*this.t,x^2*lambda^2,-alpha^2*nu*this.t*(3+nu*this.t+2*x*lambda),alpha*(-nu*this.t+2*x^2*lambda^2)]) / (2*nu*this.t*alpha^2*(alpha+1)^2);
                grad(3) = - ( 1/lambda + (alpha*nu*this.t-lambda*x)*x/(nu*this.t*alpha*(alpha+1)) );
            %else the parameters are not in the parameter space, get gradient to be zero
            else
                grad = zeros(1,3);                
            end
        end
        
    end
    
end

