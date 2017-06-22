function [ln_sum_w_ratio, y_l, y_u] = lnSumW(x, y_pow, lambda, alpha, beta)

    p = (alpha + 2) / (alpha + 1);
    phi = exp( log(1+alpha) - (2-p)*log(beta) - (p-1)*log(alpha*lambda) );

    y_max = yMax(x,phi,p);
    ln_w_max = lnWy(x,y_max,phi,p) + y_pow*log(y_max);
    
    n_term = 1E7;
    
    terms = zeros(1,n_term);
    terms(1) = 1;
    threshold = -37;
    
    got_y_l = false;
    got_y_u = false;
    
    y_l = y_max;
    y_u = y_max;
    counter = 2;
    
    if y_l == 1
        got_y_l = true;
    end
    while ~got_y_l
        y_l = y_l - 1;
        if y_l == 0
            got_y_l = true;
            y_l = y_l + 1;
        else
            log_ratio = sum([lnWy(x,y_l,phi,p), y_pow*log(y_l), -ln_w_max]);
            if log_ratio > threshold
                terms(counter) = exp(log_ratio);
                counter = counter + 1;
                if counter > n_term
                    error('Number of terms exceed memory allocation');
                end
            else
                got_y_l = true;
                y_l = y_l + 1;
            end       
        end 
    end
    
    while ~got_y_u
        y_u = y_u + 1;
        log_ratio = sum([lnWy(x,y_u,phi,p), y_pow*log(y_u), -ln_w_max]);
        if log_ratio > threshold
            terms(counter) = exp(log_ratio);
            counter = counter + 1;
            if counter > n_term
                error('Number of terms exceed memory allocation');
            end
        else
            got_y_u = true;
        end       
    end
    y_u = y_u - 1;

    counter = counter - 1;
    ln_sum_w_ratio = ln_w_max + log(sum(terms(1:counter)));

end

