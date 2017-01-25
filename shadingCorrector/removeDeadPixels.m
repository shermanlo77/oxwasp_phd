function [image, n_nan] = removeDeadPixels( image )

    nan_index = isnan(image);
    n_nan = sum(sum(nan_index));
    
    [nan_index_1, nan_index_2] = find(nan_index);
    
    for i = 1:n_nan
        
        image(nan_index_1,nan_index_2) = median([
           image(nan_index_1(i)+1,nan_index_2(i))
           image(nan_index_1(i),nan_index_2(i)+1)
           image(nan_index_1(i)-1,nan_index_2(i))
           image(nan_index_1(i),nan_index_2(i)-1)
        ]);
        
    end

end

