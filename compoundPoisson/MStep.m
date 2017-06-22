function [lambda, alpha, beta] = MStep(X, Y, alpha, beta)

    lambda = mean(Y);
    
    X = X(X~=0);
    Y = Y(Y~=0);

    theta = [alpha; beta];

    d_alpha_lnL = sum(Y*log(beta) - Y.*psi(alpha*Y) + Y.*log(X));
    d_beta_lnL = sum(Y*alpha/beta - X);

    d_alpha_beta_lnL = sum(Y/beta);

    d_alpha_alpha_lnL = -sum( exp( log(psi(1,alpha*Y))+2*log(Y) ) );
    d_beta_beta_lnL = -sum(Y*alpha/(beta^2));

    H = [d_alpha_alpha_lnL, d_alpha_beta_lnL; d_alpha_beta_lnL, d_beta_beta_lnL];
    del_lnL = [d_alpha_lnL; d_beta_lnL];

    theta = theta - H\del_lnL;

    alpha = theta(1);
    beta = theta(2);

    if any(theta<0)
        error('negative parameter');
    end

end

