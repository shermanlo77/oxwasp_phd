function cpConvergence(n_simulation, lambda, alpha, beta, n_repeat, n_step)

    lnL_array = zeros(n_step+1, n_repeat);
    lambda_array = zeros(n_step+1, n_repeat);
    alpha_array = zeros(n_step+1, n_repeat);
    beta_array = zeros(n_step+1, n_repeat);
    
    compound_poisson_true = CompoundPoisson();
    compound_poisson_true.n = n_simulation;
    compound_poisson_true.setParameters(lambda,alpha,beta);
    
    std_array = sqrt(diag(inv(compound_poisson_true.getFisherInformation())));

    for i_repeat = 1:n_repeat
        %simulated n compound poisson varibales
        X = CompoundPoisson.simulate(n_simulation,lambda,alpha,beta);
    
        %set up a compound poisson random variable
        compound_poisson = CompoundPoisson();
        compound_poisson.setParameters(lambda,alpha,beta);
        compound_poisson.addData(X);
        compound_poisson.initaliseEM();
        
        
        lnL_array(1, i_repeat) = compound_poisson.getMarginallnL();
        lambda_array(1, i_repeat) = compound_poisson.lambda;
        alpha_array(1, i_repeat) = compound_poisson.alpha;
        beta_array(1, i_repeat) = compound_poisson.beta;
        
        for i_step = 1:n_step
            compound_poisson.EStep();
            compound_poisson.MStep();
            
            lnL_array(i_step+1, i_repeat) = compound_poisson.getMarginallnL();
            lambda_array(i_step+1, i_repeat) = compound_poisson.lambda;
            alpha_array(i_step+1, i_repeat) = compound_poisson.alpha;
            beta_array(i_step+1, i_repeat) = compound_poisson.beta;
        end
        
    end
    
    figure;
    subplot(2,2,1);
    plot(0:n_step, lnL_array, 'b');
    xlabel('Number of EM steps');
    ylabel('lnL');
    subplot(2,2,2);
    plot(0:n_step, lambda_array, 'b');
    hold on;
    plot([0,n_step], [lambda-std_array(1),lambda-std_array(1)], 'k-');
    plot([0,n_step], [lambda+std_array(1),lambda+std_array(1)], 'k-');
    xlabel('Number of EM steps');
    ylabel('\lambda');
    subplot(2,2,3);
    plot(0:n_step, alpha_array, 'b');
    hold on;
    plot([0,n_step], [alpha-std_array(2),alpha-std_array(2)], 'k-');
    plot([0,n_step], [alpha+std_array(2),alpha+std_array(2)], 'k-');
    xlabel('Number of EM steps');
    ylabel('\alpha');
    subplot(2,2,4);
    plot(0:n_step, beta_array, 'b');
    hold on;
    plot([0,n_step], [beta-std_array(3),beta-std_array(3)], 'k-');
    plot([0,n_step], [beta+std_array(3),beta+std_array(3)], 'k-');
    xlabel('Number of EM steps');
    ylabel('\beta');

end

