function histogramCustom(X, edges)

    if nargin == 1
        [N, edges] = histcounts(X);
    elseif nargin == 2
        [N, edges] = histcounts(X, edges);
    end
    
    bin_width = edges(2:end) - edges(1:(end-1));
    freq_density = [0, N./bin_width, 0];
    
    stairs([edges(1),edges], freq_density);

end

