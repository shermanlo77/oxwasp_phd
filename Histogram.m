classdef Histogram < handle
  
  properties (SetAccess = protected)
    n; %count for each bin
    edges; %the border of each bin
    binWidth; %the width of each bin
    freqDensity; %freq density of each bin, count / binwidth
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    function this = Histogram(X, edges)
      if nargin == 1
        [this.n, this.edges] = histcounts(X);
      elseif nargin == 2
        [this.n, this.edges] = histcounts(X, edges);
      end
      this.binWidth = this.edges(2:end) - this.edges(1:(end-1));
      this.freqDensity = [0, this.n./this.binWidth, 0];
    end
    
    %METHOD: PLOT
    function plot(this)
      stairs([this.edges(1),this.edges], this.freqDensity);
    end
    
  end
  
end
