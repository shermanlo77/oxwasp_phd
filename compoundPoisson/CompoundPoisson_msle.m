classdef CompoundPoisson_msle < handle
    
    properties (SetAccess = private)
        %time exposure
        t;
    end
    
    methods
        
        %CONSTRUCTOR
        function this = CompoundPoisson_msle(t)
            this.t = t;
        end
        
        %SIMULATE DATA
        %PARAMETERS:
            %n: number of simulations
            %nu: poisson parameter
            %alpha: gamma shape parameter
            %lambda: gamma rate parameter
        %RETURN
            %X: row vector (size n) of simulated data
        function X = simulateData(this,n,nu,alpha,lambda)
            Y = poissrnd(nu*this.t,n,1); %simulate latent poisson variables
            X = gamrnd(Y*alpha,1/lambda); %simulate observable gamma
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
        function [f,x] = getSaddleDensity(this,x_min,x_max,n_point,nu,alpha,lambda)
            
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
        
        %ESTIMATE PARAMETERS
        %PARAMETERS:
            %X: row vector containing the data
            %initial_parameters: 3 vector containing the 3 parameters to initalise the optimisation algorithm
        %RETURN:
            %mle: 3 vector containing the maximum saddle point likelihood estimators
            %lnL: log likelihood up to a constant
        function [mle,lnL] = estimateParameters(this,X,initial_parameters)
            %define the minus log likelihood, given the data
            objective = @(parameters)this.lnL(parameters,X);
            %minimise the minus log likelihood
            [mle,lnL] = fminunc(objective,initial_parameters,optimoptions(@fminunc,'Display','off','Algorithm','quasi-newton'));
            %[mle,lnL] = fminunc(objective,initial_parameters,optimoptions(@fminunc,'Display','off','Algorithm','trust-region','GradObj','on'));
            %mle = this.stochasticGradientDescent(X,initial_parameters,10000,0.0001,0.000001);
        end

        %PLOT EMPERICAL SAMPLING DISTRIBUTION
        %PARAMETERS:
            %n_repeat: number of simulated estimators
            %n_sample: sample size of the data used for estimation
            %nu: poisson parameter
            %alpha: gamma shape parameter
            %lambda: gamma rate parameter
        function plotSamplingDistribution(this,n_repeat,n_sample,nu,alpha,lambda,n_bin)
            
            %define vector of estimators
            nu_estimate = zeros(1,n_repeat);
            alpha_estimate = zeros(1,n_repeat);
            lambda_estimate = zeros(1,n_repeat);
            
            %for n_repeat times
            for i = 1:n_repeat
                %simulate data
                X = this.simulateData(n_sample,nu,alpha,lambda);
                %get the estimator of the parameters
                %initial value on the true values
                mle = this.estimateParameters(X,[nu,alpha,lambda]);
                %save the estimators
                nu_estimate(i) = mle(1);
                alpha_estimate(i) = mle(2);
                lambda_estimate(i) = mle(3);
            end

            %plot histogram and true value of each parameter
            nestedPlot(nu_estimate,nu,'nu estimate');
            nestedPlot(alpha_estimate,alpha,'alpha estimate');
            nestedPlot(lambda_estimate,lambda,'lambda estimate');

            %NESTED FUNCTION for plotting histogram
            %PARAMETERS:
                %estimate_vector: vector of estimators
                %true_parameter: true value
                %x_axis: string for labelling the x axis
            function nestedPlot(estimate_vector,true_parameter,x_axis)
                %plot histogram
                figure;
                ax = histogram(estimate_vector,prctile(estimate_vector,linspace(0,100,n_bin+1)),'Normalization','countdensity');
                %plot true value
                hold on;
                plot([true_parameter,true_parameter],[0,max(ax.Values)],'k--','LineWidth',2);
                hold off;
                %label and set the axis
                ylabel('Frequency Density');
                xlabel(x_axis);
                xlim([min(prctile(estimate_vector,5),true_parameter),max(prctile(estimate_vector,95),true_parameter)]);
            end
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
            if nargin > 1
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
        
        %PLOT SIMULATION
        %For given parameters, plot saddle density and histogram of simulation
        %PARAMETERS:
            %nu: poisson parameter
            %alpha: gamma shape parameter
            %lambda: gamma rate parameter
            %n_bins: number of bins for the histogram
        function X = plotSimulation(this,n,nu,alpha,lambda,n_bins)
            
            %simulate the data
            X = this.simulateData(n,nu,alpha,lambda);
            %from the data, get the edges of the histogram
            edges = prctile(X,linspace(0,100,n_bins+1));
            
            %x_min is the minimum value to plot the saddle density
            %if there is a zero value
            if any(X==0)
                %x_min is a ratio to the maximum value
                x_min = max(X)/1000;
            else
                %else x_min is the minimum value
                x_min = min(X);
            end
            
            %get the saddle density
            [f,x] = this.getSaddleDensity(x_min,max(X),1000,nu,alpha,lambda);
            
            %plot the histogram
            figure;
            histogram(X,edges,'Normalization','pdf');
            %plot the saddle density
            hold on;
            plot(x,f);
            hold off;
            %set and label the axis
            xlabel('Support');
            ylabel('p.d.f.');
            legend('Exact simulation','Saddlepoint approx.');
        end
        
        %STOCHASTIC GRADIENT DESCENT
        %PARAMETERS:
            %X: row vector of data
            %intial: 3 row vector of the initial parameters
            %n_step: number of steps
            %step_size: step size of each stop in stochastic gradient descent
            %tolerance: how much as a ratio the parameter can change before stopping the algorithm
        %RETURN:
            %mle: 3 row vector for the optimised parameter
            %mle_path: design matrix transpose (3 x n_step+1) form of the mle at each step of stochastic gradient descent
        function [mle,mle_path] = stochasticGradientDescent(this,X,initial,n_step,step_size,tolerance)
            
            %mle_path is a (3 x n_step+1), each column is the mle at each step
            mle_path = zeros(3,n_step+1);
            %on the zeroth step, the mle is the initial value
            mle_path(:,1) = initial';
            mle = initial;
            %tolerance_met is false, becomes true when the parameters doesn't change that much compared to tolerance
            tolerance_met = false;
            
            %for n_step times
            for i = 1:n_step
                %if the tolerance hasn't been met, update the mle
                if ~tolerance_met
                    %get a datapoint in sequence
                    x = X(mod(i-1,numel(X))+1);
                    %do stochastic gradient descent
                    mle_new = mle - step_size * this.gradient(mle,x);
                    %if all the parameters are positive
                    if all(mle_new>0)
                        %check if the tolerance has been met
                        if all(abs(1-mle_new./mle))<tolerance
                            %if so, set tolerance_met = true
                            tolerance_met = true;
                        end
                        %update the mle
                        mle = mle_new;
                    end
                end
                %save the mle to mle_path
                mle_path(:,i+1) = mle';
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
                term_3 = exp(sum([log(nu*this.t)/(alpha+1),(alpha/(alpha+1))*sum([log(lambda),log(x),-log(alpha)]),-log(nu)]));
                del_1 = sum([term_1,term_2,term_3]);
            end

            %NESTED FUNCTION for gradient with respect to alpha
            function del_2 = grad_2()
                term_1 = exp(sum([log(2),log(alpha),log(alpha+1),log(nu*this.t)/(1+alpha),(alpha/(alpha+1))*sum([log(lambda),log(x),-log(alpha)])]));
                term_1 = term_1 * sum([log(x),log(nu*this.t),-log(alpha)]);
                term_2 = alpha*sum([log(alpha),alpha*log(lambda),log(nu*this.t)]);
                del_2 = sum([term_1,1,-alpha^2,term_2,alpha*(1+alpha)*log(lambda),alpha*log(x)]);
                del_2 = del_2 / (2*alpha*(alpha+1)^2);
            end

            %NESTED FUNCTION for gradient with respect to lambda
            function del_3 = grad_3()
                del_3 = sum([alpha/(2*(alpha+1)*lambda),-x,(nu*this.t*alpha/lambda)^(1/(alpha+1))*x^(alpha/(alpha+1))]);
            end
        
        end
        
        
    end
    
end

