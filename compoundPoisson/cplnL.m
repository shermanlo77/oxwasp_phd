function lnL = cplnL(X, lambda, alpha, beta)
    
    lnL_terms = zeros(1,numel(X));
    
    for i = 1:numel(X)

        x = X(i);
        
        terms = zeros(1,3);
        terms(1) = -lambda-x*beta;
        
        if x>0
            terms(2) = -log(x);
            terms(3) = lnSumW(x, 0, lambda, alpha, beta);
        end

        lnL_terms(i) = sum(terms);
        
    end

    lnL = sum(lnL_terms);

end

