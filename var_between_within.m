function [var_b, var_w] = var_between_within(image_stack)

    [height, width, n] = size(image_stack);
    area = height*width;
    
    mean_image = mean(image_stack,3);
    mean_all = mean(reshape(mean_image,[],1));

    var_w = sum(sum(sum( ( image_stack - repmat(mean_image,1,1,n) ).^2 ))) / (area*n - area);
    %save the between pixel variance
    var_b = n * sum(sum((mean_image - mean_all).^2))/(area-1);

end

