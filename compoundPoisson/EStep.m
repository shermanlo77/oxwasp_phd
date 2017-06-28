function [expectation,variance] = EStep(x, lambda, alpha, beta)

    if x == 0
        expectation = 0;
        variance = nan;
    else
        normalisation_constant = lnSumW(x, 0, lambda, alpha, beta);
        expectation = exp(lnSumW(x, 1, lambda, alpha, beta) - normalisation_constant);
        variance = exp(lnSumW(x, 2, lambda, alpha, beta) - normalisation_constant) - expectation^2;
    end


end

