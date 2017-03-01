%CHI2GOF_NORM
%Chi squared goodness of fit test for the Normal distribution
%PARAMETERS:
    %x: vector of data
    %f_e: expected frequency of each bin
    %is_need_estimation: boolean, true to estimate the mean and standard deviation, loses 2 degrees of freedom, if false assume the target distribution is N(0,1)
%RETURN
    %p_value: a single chi squared p value
    %chi_squared: array of chi squared statistic from each bin
    %edges: array of the edges of each bin
function [p_value, chi_squared, edges] = chi2gof_norm(x, f_e, is_need_estimation)

    %if the mean and standard deviation needs estimation
    if is_need_estimation
        %estimate the mean and standard deviation
        mu = mean(x);
        sigma = std(x);
        %normalise the 
        x = (x - mu) ./ sigma;
    end

    %get the number of data
    n = numel(x);
    %get the number of bins for a given expected frequency
    n_bin = round(n/f_e);
    %get the degrees of freedom
    dof = n_bin - is_need_estimation*2 - 1;
    %if the degrees of freedom is too small, throw error
    if dof < 1
        error('Not enough bins');
    end
    
    %get the probability a data point will be in a bin
    p_length = f_e/n;
    
    %declare array of edges
    edges = zeros(n_bin+1,1);
    %the first edge is at -infinity
    edges(1) = -inf;
    %for each bin
    for i = 1:n_bin
        %get the edge of the bin and save it
        %the edges are defined to have equal probabiltiy in each bin
        edges(i+1) = norminv(i*p_length);
    end

    %declare array of observed frequency in each bin
    f_o_array = zeros(n_bin,1);
    
    %for each bin
    for i = 1:n_bin
        %get the observed frequency
        f_o_array(i) = sum( (edges(i) < x) & (x < edges(i+1)) );
    end
    
    %work out the chi squared statistic
    chi_squared = (f_o_array - f_e).^2 / f_e ;
    
    %get the right hand side cdf of the chi squared statistic
    p_value = chi2cdf(sum(chi_squared), dof, 'upper');

end
