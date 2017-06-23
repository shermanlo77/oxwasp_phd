function cp_pdf = cpPdf(x, lambda, alpha, beta)

    cp_pdf = exp(cplnL(x, lambda, alpha, beta));

end

