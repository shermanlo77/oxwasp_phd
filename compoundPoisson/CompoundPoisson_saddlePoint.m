classdef CompoundPoisson_saddlePoint < CompoundPoisson
    
    methods
        
        %CONSTRUCTOR
        function this = CompoundPoisson_saddlePoint(t)
            this = this@CompoundPoisson(t);
        end
 
        %GET SADDLEPOINT DENSITY
        %For a range of the domain and given parameters, return the
        %saddlepoint density. The normalisation constant was worked out
        %using the trapezium rule, with n_point - 1 strips
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
            k = -nu; %some constant to control over/under flow
            %work out the saddle density
            f = exp(k+sum([-(alpha+2)/(2*(alpha+1))*log(x);-x*lambda;(x/alpha).^(alpha/(alpha+1))*(nu*this.t)^(1/(alpha+1))*(lambda)^(alpha/(alpha+1))*(alpha+1)]));

            %work out the height of the trapziums
            h = (x_max-x_min)/(n_point-1);
            %integrate the function
            area = 0.5*h*(f(1)+f(end)+2*sum(f(2:(end-1))));
            
            %normalise the saddle density
            f = f/area;

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
            %for x=0, set it to be the minimum of x / 2
            X(X==0) = min(X(X~=0))/2;

            %for valid values of the parameters
            if((nu>0)&&(this.t>0)&&(alpha>0)&&(lambda>0))
                
                %work out the minus log likelihood

                log_likelihood = -sum(sum([
                    -(alpha+2)/(2*(alpha+1))*log(X),-lambda*X,(X*lambda/alpha).^(alpha/(alpha+1))*(nu*this.t)^(1/(alpha+1))*(alpha+1)
                    ]));

                log_likelihood = log_likelihood + n*log(alpha+1)/2;
                log_likelihood = log_likelihood + n*nu*this.t;
                log_likelihood = log_likelihood - n/2/(alpha+1)*(sum([log(nu),log(this.t),alpha*log(lambda),log(alpha)]));

            %else return the maximum value for the minus log likelihood
            else
                log_likelihood = inf;
            end
            
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
        %Evulates the gradient of -lnL for a given single datapoint
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
                grad(1) = -grad_1();
                grad(2) = -grad_2();
                grad(3) = -grad_3();
            %else the parameters are not in the parameter space, get gradient to be zero
            else
                grad = zeros(1,3);                
            end
            
            %NESTED FUNCTION for gradient with respect to nu
            function del_1 = grad_1()
                term_1 = 1/(2*(alpha+1)*nu);
                term_2 = -this.t;
                term_3 = exp(sum([log(this.t)/(alpha+1),(alpha/(alpha+1))*sum([log(lambda),log(x),-log(alpha),-log(nu)])]));
                del_1 = sum([term_1,term_2,term_3]);
            end

            %NESTED FUNCTION for gradient with respect to alpha
            function del_2 = grad_2()
                term_1 = exp(sum([log(2),log(alpha),log(alpha+1),log(nu*this.t)/(1+alpha),(alpha/(alpha+1))*sum([log(lambda),log(x),-log(alpha)])]));
                term_1 = term_1 * sum([log(x),log(lambda),-log(nu*this.t),-log(alpha)]);
                term_2 = -alpha*sum([log(alpha),alpha*log(lambda),log(nu*this.t)]);
                del_2 = sum([term_1,1,-alpha^2,term_2,alpha*(1+alpha)*log(lambda),alpha*log(x)]);
                del_2 = del_2 / (2*alpha*(alpha+1)^2);
            end

            %NESTED FUNCTION for gradient with respect to lambda
            function del_3 = grad_3()
                term_1 = alpha/(2*(alpha+1)*lambda);
                term_2 = exp(sum([sum([log(nu*this.t),log(alpha),-log(lambda)])/(alpha+1),(alpha/(alpha+1))*log(x)]));
                del_3 = sum([term_1,-x,term_2]);
            end
        
        end
        
        
    end
    
end

