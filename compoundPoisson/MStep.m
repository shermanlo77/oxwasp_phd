function [lambda, alpha, beta] = MStep(X, Y, var, alpha, beta)

    lambda = mean(Y);
    
    X = X(X~=0);
    Y = Y(Y~=0);
    var = var(~isnan(var));

    theta = [alpha; beta];

    d_alpha_lnL = sum(Y*log(beta) - Y.*psi(alpha*Y) + Y.*log(X) - (0.5*alpha^2).*var.*Y.*psi(2,Y*alpha) - alpha*var.*psi(1,Y*alpha));
    d_beta_lnL = sum(Y*alpha/beta - X);

    d_alpha_beta_lnL = sum(Y/beta);

    d_alpha_alpha_lnL = -sum( Y.^2.*psi(1,alpha*Y) + var.*psi(1,alpha*Y) + (2*alpha).*var.*Y.*psi(2,alpha*Y) + (0.5*alpha^2).*var.*Y.^2.*psi(3,Y*alpha) );
    d_beta_beta_lnL = -sum(Y*alpha/(beta^2));

    H = [d_alpha_alpha_lnL, d_alpha_beta_lnL; d_alpha_beta_lnL, d_beta_beta_lnL];
    del_lnL = [d_alpha_lnL; d_beta_lnL];

    theta = theta - H\del_lnL;

    if any(theta<0)
        error('negative parameter');
    end
    
    alpha = theta(1);
    beta = theta(2);

    

end

