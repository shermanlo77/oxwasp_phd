function lnL = cpFulllnL(X, Y, lambda, alpha, beta )
    
    n = numel(X);
    
    lnL = zeros(7,n);
    
    for i = 1:n
        x = X(i);
        y = Y(i);
        
        if x > 0
            lnL(1,i) = y*alpha*log(beta);
            lnL(2,i) = -gammaln(y*alpha);
            lnL(3,i) = y*alpha*log(x);
            lnL(4,i) = -x*beta;
            lnL(7,i) = -0.5*y*alpha^2*psi(1,y*alpha);
        end
        
        lnL(5,i) = y*log(lambda);
        lnL(6,i) = -lambda;
    end
    
    lnL = sum(sum(lnL));


end

