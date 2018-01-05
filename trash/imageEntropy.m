%FUNCTION: IMAGE ENTROPY
%Calculates the imperical entropy, this is done by extracting the greyvalues from the image
%each greyvalue is rounded to an integer
%S = sum(-p*log(p)) over all integer greyvalues
%p is estimated by the number of pixels with that greyvalue divide by the number of pixels 
%PARAMETERS:
    %image: matrix of greyvalues
    %base: 2 or 10, returns the entropy using the corresponding base for the logarithm
%RETURNS:
    %S: entropy
function S = imageEntropy(image, base)

    %get the vector of greyvalues
    x = round(reshape(image,[],1));
    %get the number of pixels
    area = numel(x);
    
    %get the minimum and maximum greyvalue
    min_x = min(x);
    max_x = max(x);
    
    %declare an array, one element for each greyvalue between min_x and max_x
    %this will count the number of pixels which has a certain greyvalue
    n_array = zeros(max_x - min_x + 1,1);
    
    %for each pixel
    for i = 1:area
        %get the greyvalue
        x_i = x(i);
        %if that greyvalue is a number
        if ~isnan(x_i)
            %increment the corresponding element of n_array by one
            n_array(x_i - min_x + 1) = n_array(x_i - min_x + 1) + 1;
        else
            %else that greyvalue is not a number, ignore and adjust area accordingly
            area = area - 1;
        end
    end
    
    %turn n_array which counts, into a probability
    n_array = n_array / area;
    %any zero values, covert it to 1 so that the entropy calculation returns 0 instead of nan
    n_array(n_array==0) = 1;
    
    %work out the entropy using the corresponding base
    if base == 10
        S = -sum(n_array.*log(n_array));
    elseif base == 2
        S = -sum(n_array.*log2(n_array));
    end
    
end

