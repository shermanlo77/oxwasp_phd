%QUOTE QUARTILE ERROR
%Returns the quartiles in the form of error bars and the order of
%magnitude, the number of significant figures is justified by using the
%order of magnitude of the bootstrap error of the error
%PARAMETERS:
    %data: vector of data
    %n_bootstrap: number of boostrap samples to use
%RETURN:
    %q2: median
    %up_error
    %down_error
    %E: exponent
    %sig_fig: number of significant figures of q2
    %error_sig_fig: number of significant figures of the error
function [q2, up_error, down_error, E, sig_fig, error_sig_fig] = quoteQuartileError(data, n_bootstrap)

    %get the number of data
    n = numel(data);

    %work out the quantiles
    q_array = quantile(data,[0.25,0.5,0.75]);
    %get the median
    q2 = q_array(2);
    %get the difference between Q3 and Q1 with the median
    up_error = q_array(3) - q2;
    down_error = q2 - q_array(1);
    
    %declare array of bootstrap samples of the up/down error
    %up/down error is the difference between Q3 and Q1 with the median
    up_error_bootstrap = zeros(n_bootstrap,1);
    down_error_bootstrap = zeros(n_bootstrap,1);
    
    %for n_boostrap times
    for i_bootstrap = 1:n_bootstrap
        
        %get the bootstrap sample of the data
        bootstrap_index = randi([1,n],n,1);
        data_bootstrap = data(bootstrap_index);
        
        %get the up/down error and save it to the array
        q_array = quantile(data_bootstrap,[0.25,0.5,0.75]);
        up_error_bootstrap(i_bootstrap) = q_array(3) - q_array(2);
        down_error_bootstrap(i_bootstrap) = q_array(2) - q_array(1);
        
    end
    
    %get the minus minimum order of magnitude of the bootstrap samples of the up/down error, add 1
    %it will be used as the number of significant figures to round the error
    error_sig_fig = -min([orderMagnitude(std(up_error_bootstrap)),orderMagnitude(std(down_error_bootstrap))]) + 1;
    
    %error_sig_fig is an integer, if it is equal to 0 or less
    if error_sig_fig <= 0
        error_sig_fig = 1;
    end
    %else the error doesn't vary much, leave it as the number of significant figures
    
    %round the error using error_sig_fig number of significant figures
    up_error = round(up_error,error_sig_fig,'significant');
    down_error = round(down_error,error_sig_fig,'significant');

    %E is the exponent of q2
    E = floor(log10(abs(q2)));
    
    %get the mantissa of the errors and q2
    up_error = up_error / 10^E;
    down_error = down_error / 10^E;
    q2 = q2 / 10^E;
    
    %get the number of decimial places of the least significant figure of the error
    dec_places = -floor(min(log10([up_error,down_error]))) + error_sig_fig - 1;
    %if it is less or equal to 0, set signiciant figures to 1
    if dec_places <= 0
        sig_fig = 1;
    %else, set the number of signiciant figures to the number of decimial places add 1
    else
        sig_fig = dec_places + 1;
    end
    
    %round q2
    q2 = round(q2,sig_fig,'significant');
end

