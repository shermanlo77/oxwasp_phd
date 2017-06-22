function expectation = EStep(x, lambda, alpha, beta)

    if x == 0
        expectation = 0;
    else
        expectation = exp(lnSumW(x, 1, lambda, alpha, beta) - lnSumW(x, 0, lambda, alpha, beta));
    end


end

