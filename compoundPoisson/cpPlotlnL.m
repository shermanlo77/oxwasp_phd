%FUNCTION: COMPOUND POISSON PLOT LOG LIKELIHOOD
%Plots the log likelihood as a function of alpha and beta
%PARAMETERS:
    %compound_poisson: compound poisson object with data and parameters
    %alpha: 2 vector, alpha values
    %beta: 2 vector, beta values
    %n: number of points in an axis
function cpPlotlnL(compound_poisson, alpha, beta, n)

    %get a mesh grid of alphas and betas
    [alpha_grid, beta_grid] = meshgrid(linspace(alpha(1),alpha(2),n),linspace(beta(1),beta(2),n));
    %get a mesh grid of the log likelihoods
    lnL_full_grid = alpha_grid;
    %for each value in the grid
    for i = 1:numel(lnL_full_grid)
        %set the compound poisson to have that parameter
        compound_poisson.setParameters(compound_poisson.lambda,alpha_grid(i),beta_grid(i));
        %get the log likelihood
        lnL_full_grid(i) = compound_poisson.getMarginallnL();
    end
    
    %surf plot the log likelihood
    figure;
    surf(alpha_grid, beta_grid, lnL_full_grid);
    xlabel('\alpha');
    ylabel('\beta');
    zlabel('log likelihood');

end

