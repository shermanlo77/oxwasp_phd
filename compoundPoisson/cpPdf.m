function cp_pdf = cpPdf(x, lambda, alpha, gamma)

    p = (alpha + 2) / (alpha + 1);
    phi = exp( log(1+alpha) + (2-p)*log(gamma) - (p-1)*log(alpha*lambda) );
    
    terms = zeros(1,3);
    terms(1) = -lambda-x/gamma;
    terms(2) = -log(x);
    terms(3) = lnSumW(x,phi,p);

    cp_pdf = exp(sum(terms));


end

