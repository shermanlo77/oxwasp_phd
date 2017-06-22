function cp_pdf = cpPdf(x, lambda, alpha, beta)
    
    terms = zeros(1,3);
    terms(1) = -lambda-x*beta;
    terms(2) = -log(x);
    terms(3) = lnSumW(x, 0, lambda, alpha, beta);

    cp_pdf = exp(sum(terms));


end

