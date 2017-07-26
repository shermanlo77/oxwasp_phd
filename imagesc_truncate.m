%FUNCTION: IMAGE SCALE TRUNCATE
%Heatmap plot the image X, the scale is adjusted to remove outliers.
%Outliers are bigger than Q3+1.5*IQR or smaller than Q1-1.5*IQR
%PARAMETERS:
    %X: matrix of values
function ax = imagesc_truncate(X)

    %get the 25th and 75th percentile
    q = prctile(reshape(X,[],1),[25,75]);
    %work out the inter quantile range
    iqr = q(2) - q(1);
    
    %work out the upper limit of the image value
    x_upper = q(2) + 1.5*iqr;
    %work out the lower limit of the image value
    x_lower = q(1) - 1.5*iqr;

    %put the limits in a vector
    clims = [x_lower,x_upper];
    
    %plot image with adjusted scale
    ax = axes;
    imagesc(X,clims);
    
end

