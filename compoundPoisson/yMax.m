function y_max = yMax(x, phi, p)
    
    y_max = round(exp( (2-p)*log(x) - log(phi) - log(2-p) ));
    
    if y_max == 0
        y_max = 1;
    end

end

