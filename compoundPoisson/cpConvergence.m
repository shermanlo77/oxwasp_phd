%FUNCTION: COMPOUND POISSON CONVERGENCE
%Plots the log likelihood and parameters at each step of EM
%The starting points for the EM algorithm is at the true value
%PARAMETERS:
    %n_simulation: number of data points in the simulation
    %lambda: poisson parameter
    %alpha: gamma shape parameter
    %beta: gamma rate parameter
    %n_repeat: number of times to repeat the experiment
    %n_step: number of EM steps
function cpConvergence(n_simulation, lambda, alpha, beta, n_repeat, n_step)

    %declare array of lnL, lambda, alpha and beta for each step of EM and each repeat
        %dim 1: for each step of EM
        %dim 2: for each repeat of the experiment
    lnL_array = zeros(n_step+1, n_repeat);
    lambda_array = zeros(n_step+1, n_repeat);
    alpha_array = zeros(n_step+1, n_repeat);
    beta_array = zeros(n_step+1, n_repeat);
    
    %instantise a compound Poisson with the true parameters
    compound_poisson_true = CompoundPoisson();
    compound_poisson_true.n = n_simulation;
    compound_poisson_true.setParameters(lambda,alpha,beta);
    
    %get the standard error of the estimators of the 3 parameters using the Fisher's information matrix
    std_array = sqrt(diag(inv(compound_poisson_true.getFisherInformation())));

    %for n_repeat times
    for i_repeat = 1:n_repeat
        
        %simulate n compound poisson varibales
        X = CompoundPoisson.simulate(n_simulation,lambda,alpha,beta);
    
        %set up a compound poisson random variable
        compound_poisson = CompoundPoisson();
        compound_poisson.setParameters(lambda,alpha,beta);
        compound_poisson.addData(X);
        compound_poisson.initaliseEM();
        
        %save the log likelihood, lambda, alpha and beta before EM
        lnL_array(1, i_repeat) = compound_poisson.getMarginallnL();
        lambda_array(1, i_repeat) = compound_poisson.lambda;
        alpha_array(1, i_repeat) = compound_poisson.alpha;
        beta_array(1, i_repeat) = compound_poisson.beta;
        
        %for n_step times
        for i_step = 1:n_step
            %take a E and M step
            compound_poisson.EStep();
            compound_poisson.MStep();
            %save the log likelihood, lambda, alpha and beta before EM
            lnL_array(i_step+1, i_repeat) = compound_poisson.getMarginallnL();
            lambda_array(i_step+1, i_repeat) = compound_poisson.lambda;
            alpha_array(i_step+1, i_repeat) = compound_poisson.alpha;
            beta_array(i_step+1, i_repeat) = compound_poisson.beta;
        end
    end
    
    
    figure;
    
    %plot the log likelihood
    subplot(2,2,1);
    plot(0:n_step, lnL_array, 'b');
    xlabel('Number of EM steps');
    ylabel('lnL');
    xlim([0,n_step]);
    
    %plot lambda
    subplot(2,2,2);
    plot(0:n_step, lambda_array, 'b');
    hold on;
    plot([0,n_step], [lambda-std_array(1),lambda-std_array(1)], 'k-');
    plot([0,n_step], [lambda+std_array(1),lambda+std_array(1)], 'k-');
    xlabel('Number of EM steps');
    ylabel('\lambda');
    xlim([0,n_step]);
    
    %plot alpha
    subplot(2,2,3);
    plot(0:n_step, alpha_array, 'b');
    hold on;
    plot([0,n_step], [alpha-std_array(2),alpha-std_array(2)], 'k-');
    plot([0,n_step], [alpha+std_array(2),alpha+std_array(2)], 'k-');
    xlabel('Number of EM steps');
    ylabel('\alpha');
    xlim([0,n_step]);
    
    %plot beta
    subplot(2,2,4);
    plot(0:n_step, beta_array, 'b');
    hold on;
    plot([0,n_step], [beta-std_array(3),beta-std_array(3)], 'k-');
    plot([0,n_step], [beta+std_array(3),beta+std_array(3)], 'k-');
    xlabel('Number of EM steps');
    ylabel('\beta');
    xlim([0,n_step]);

end

