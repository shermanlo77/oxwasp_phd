%QUOTE QUARTILE ERROR
%Returns the quartiles in the form of error bars and the order of
%magnitude, the number of significant figures is justified by using the
%order of magnitude of the bootstrap error of the error
%PARAMETERS:
    %data: vector of data
    %n_bootstrap: number of boostrap samples to use, if 1 use 1 sig fig, if
    %2 use 2 sig fig for the error
%RETURN:
    %quote: string
function quote = quoteQuartileError(data, n_bootstrap)

    %get the number of data
    n = numel(data);

    %work out the quantiles
    q_array = quantile(data,[0.25,0.5,0.75]);
    %get the median
    q2 = q_array(2);
    %get the difference between Q3 and Q1 with the median
    up_error = q_array(3) - q2;
    down_error = q2 - q_array(1);
    
    %set the number of significant figures to be used for the error
    if n_bootstrap == 1
        error_sig_fig = 1;
    elseif n_bootstrap == 2
        error_sig_fig = 2;
    %else, use bootstrap samples to see how much the error varies
    else
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

        %get the minimum order of magnitude of the ratio between the error and the bootstrap variance
        error_mag_order = min([orderMagnitude(std(up_error_bootstrap)/up_error),orderMagnitude(std(down_error_bootstrap)/down_error)]);

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
    E = floor(log10(abs(q2)));
    
    %get the mantissa of the errors and q2
    up_error = up_error * 10^-E;
    down_error = down_error * 10^-E;
    q2 = q2 * 10^-E;
    
    %get the number of decimial places of the least significant figure of the error
        %-floor(min(log10([up_error,down_error]))) gets the minus exponent of the errors
        %error_sig_fig - 1 increases the number of decimial places according to the number of significant figures of the error
    dec_places = -floor(min(log10([up_error,down_error]))) + error_sig_fig - 1;
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
    up_error = round(up_error,dec_places,'decimals');
    down_error = round(down_error,dec_places,'decimals');
    
    %round q2
    q2 = round(q2,sig_fig,'significant');
    
    %convert the error to string
    up_error = num2str(up_error);
    down_error = num2str(down_error);
    
    %fill in missing decimial places with zeros
    if dec_places ~= 0
        while numel(up_error)<2+dec_places
            up_error = [up_error,'0'];
        end
        while numel(down_error)<2+dec_places
            down_error = [down_error,'0'];
        end
    end
    
    %convert the exponent to string
    E = num2str(E);
    
    %convert q2 to string
    if sig_fig == 1
        q2 = num2str(q2);
    else
        q2 = num2str(q2 * 10^(sig_fig-1));
        q2 = strcat(q2(1),'.',q2(2:end));
    end
    
    %export the quote value as a string
    if E == '0'
        quote = strcat('$',q2,'\substack{+',up_error,'\\ -',down_error,'}$');
    else %put brackets around the value in scientific notation
        quote = strcat('$(',q2,'\substack{+',up_error,'\\ -',down_error,'})\times 10^{',E,'}$');
    end
        
end

