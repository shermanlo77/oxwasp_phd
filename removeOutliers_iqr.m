function index = removeOutliers_iqr(x)

    %get q1 and q3
    q = prctile(x,[25,75]);
    q1 = q(1);
    q2 = q(2);
    %work out iqr
    iqr = q2-q1;
    %find the index which point to data which are not outliers
    index =  ( x > (q1 - 1.5*iqr) );
    index = index & ( x < (q2 + 1.5*iqr) );

end

