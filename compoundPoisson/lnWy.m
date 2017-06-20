function ln_Wy = lnWy(x, y, phi, p)

    alpha = (2-p)/(1-p); %negative value

    terms = zeros(1,6);
    
    terms(1) = y*alpha*log(p-1);
    terms(2) = -y*alpha*log(x);
    terms(3) = -y*(1-alpha)*log(phi);
    terms(4) = -y*log(2-p);
    terms(5) = -gammaln(1+y);
    terms(6) = -gamma(-alpha*y);
    
    ln_Wy = sum(terms);


end

