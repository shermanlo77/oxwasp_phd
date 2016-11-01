function log_likelihood = lnL(parameters,t,X)

    nu = parameters(1);
    alpha = parameters(2);
    lambda = parameters(3);

    n = numel(X);
    X(X==0) = 0.0000000000001;

    if((nu>0)&&(t>0)&&(alpha>0)&&(lambda>0))

        log_likelihood = -sum(sum([
            -(alpha+2)/(2*(alpha+1))*log(X),-lambda*X,(X*lambda/alpha).^(alpha/(alpha+1))*(nu*t)^(1/(alpha+1))*(alpha+1)
            ]));
        
        log_likelihood = log_likelihood + n*log(alpha+1)/2;
        log_likelihood = log_likelihood + n*nu*t;
        log_likelihood = log_likelihood - n/2/(alpha+1)*(sum([log(nu),log(t),alpha*log(lambda),log(alpha)]));
        
    else
        log_likelihood = inf;
    end


end

