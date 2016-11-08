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
        function log_likelihood = lnL(this,parameters,X)

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
        end
        
        %PLOT SIMULATION
        %For given parameters, plot saddle density and histogram of simulation
        %PARAMETERS:
            %nu: poisson parameter
            %alpha: gamma shape parameter
            %lambda: gamma rate parameter
            %n_bins: number of bins for the histogram
        function plotSimulation(this,nu,alpha,lambda,n_bins)
            
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
        
        
    end
    
end

