classdef CompoundPoisson < handle
    
    %COMPOUND POISSON ABSTRACT SUPER CLASS
    %Abstract class modelling the compound poisson gamma distribution. Each
    %subclass implements its own approximation for the density and log
    %likelihood
    
    %MEMBER VARIABLES
    properties (SetAccess = protected)
        %time exposure
        t;
    end
    
    %ABSTRACT METHODS
    methods (Abstract)
        
        %GET DENSITY
        %For a range of the domain and given parameters, return the density.
        %PARAMETERS:
            %x_min: lowest part of the domain
            %x_max: highest part of the domain
            %n_point: number of points between x_min and x_max
            %nu: poisson parameter
            %alpha: gamma shape parameter
            %lambda: gamma rate parameter
        %RETURN:
            %f: row vector of size n_points containing the density for each point
            %x: row vector of size n_points, linspace(x_min,x_max,n_point)
        [f,x] = getDensity(this,x_min,x_max,n_point,nu,alpha,lambda);
        
        %MINUS LOG LIKELIHOOD
        %PARAMETERS:
            %parameters: 3 vector containing the parameters
            %X: row vector containing data
        %RETURN
            %log_likelihood: -lnL
            %grad: gradient
        [log_likelihood, grad] = lnL(this,parameters,X);
        
        %GRADIENT
        %Evaluates the gradient of -lnL for a given single datapoint
        %PARAMETERS:
            %parameter: 3 row vector containing the parameters
            %x: datapoint
        grad = gradient(this,parameter,x)
        
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        function this = CompoundPoisson(t)
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
                %ax = histogram(estimate_vector,'Normalization','countdensity');
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
            [f,x] = this.getDensity(x_min,max(X),1000,nu,alpha,lambda);
            
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
            legend('Exact simulation','Approx. density');
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
        
        function estimator = methodOfMoments(this,X)
            mu_1 = mean(X);
            mu_2 = moment(X,2);
            mu_3 = moment(X,3);

            psi_1 = mu_2/mu_1;
            psi_2 = mu_3/mu_2;
            
            estimator = zeros(1,3);

            estimator(1) = mu_1/(2*psi_1-psi_2)/this.t;
            estimator(2) = (2*psi_1-psi_2)/(psi_2-psi_1);
            estimator(3) = 1/(psi_2-psi_1);
        end
        
    end
    
end

