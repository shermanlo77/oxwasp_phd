function ax = histogram_constFreq(x, n_bin)
    edges = prctile(x,linspace(0,100,n_bin+1));
    ax = histogram(x, edges, 'Normalization', 'countdensity');
end

