%QUOTE STD ERROR
%Returns the mean and std error bar in the form of error bars and the order of
%magnitude, the number of significant figures is justified by using the
%order of magnitude of the bootstrap error of the error
%PARAMETERS:
    %data: vector of data
    %n_bootstrap: number of boostrap samples to use, if 1 use 1 sig fig, if
    %2 use 2 sig fig for the error
%RETURN:
    %quote: string
function quote = quoteStdError(data, n_bootstrap)

    %get the number of data
    n = numel(data);

    %work out the mean
    q = mean(data);
    %get the std
    err = std(data);
    
    %set the number of significant figures to be used for the error
    if n_bootstrap == 1
        error_sig_fig = 1;
    elseif n_bootstrap == 2
        error_sig_fig = 2;
    %else, use bootstrap samples to see how much the error varies
    else
        %declare array of bootstrap samples of the error
        error_bootstrap = zeros(n_bootstrap,1);
        %for n_boostrap times
        for i_bootstrap = 1:n_bootstrap
            %get the bootstrap sample of the data
            bootstrap_index = randi([1,n],n,1);
            data_bootstrap = data(bootstrap_index);
            %estimate the error of the bootstrap sample
            error_bootstrap(i_bootstrap) = std(data_bootstrap);
        end

        %get the order of magnitude of the ratio between the error and the bootstrap variance
        error_mag_order = orderMagnitude(std(error_bootstrap)/err);

        %error_sig_fig is an integer, if it is equal to 0 or more
        if error_mag_order >= 0
            error_sig_fig = 1;
        %else the number of significant figures is 2 or -error_mag_order+1
        else
            error_sig_fig = 2;
            %error_sig_fig = -error_mag_order+1;
        end
    end

    %E is the exponent of q2
    E = floor(log10(abs(q)));
    
    %get the mantissa of the errors and q2
    err = err * 10^-E;
    q = q * 10^-E;
    
    %get the number of decimial places of the least significant figure of the error
        %-floor(log10(err)) gets the exponent of the errors
        %error_sig_fig - 1 increases the number of decimial places according to the number of significant figures of the error
    dec_places = -floor(log10(err)) + error_sig_fig - 1;
    %if it is less or equal to 0, set significant figures to 1
    if dec_places <= 0
        sig_fig = 1;
        dec_places = 0;
    %else, set the number of signifiant figures to the number of decimial places add 1
        %add one for the digit to the left of the decimial place
    else
        sig_fig = dec_places + 1;
    end
    
    %round the error using dec_places number of decimial places
    err = round(err,dec_places,'decimals');
    
    %round q2
    q = round(q,sig_fig,'significant');
    
    %convert the error to string
    err = num2str(err);
    
    %fill in missing decimial places with zeros
    if dec_places ~= 0
        while numel(err)<2+dec_places
            err = [err,'0'];
        end
    end
    
    %convert the exponent to string
    E = num2str(E);
    
    %convert q to string
    if sig_fig == 1
        q = num2str(q);
    else
        q = num2str(q * 10^(sig_fig-1));
        q = strcat(q(1),'.',q(2:end));
    end
    
    %export the quote value as a string
    if E == '0'
        quote = strcat('$',q,'\pm',err,'$');
    else %put brackets around the value in scientific notation
        quote = strcat('$(',q,'\pm',err,')\times 10^{',E,'}$');
    end
        
end

