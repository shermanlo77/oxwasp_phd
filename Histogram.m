%MIT License
%Copyright (c) 2019 Sherman Lo

%CLASS: HISTOGRAM
%A custom histogram class
%
%Plots the histogram (frequency density) as a line function
%How to use:
  %pass the data, and optionally the bin edges, via the constructor
  %class the method plot
  %properties of the histogram can be extracted from the member variables
classdef Histogram < handle
  
  properties (SetAccess = protected)
    n; %count for each bin
    edges; %the border of each bin
    binWidth; %the width of each bin
    freqDensity; %freq density of each bin, count / binwidth
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    %PARAMETERS:
      %X: vector of data
      %edges (optional): the edges of the bins, vector size numel(X)+1, if empty, uses bins in
          %the default MATLAB function histcounts
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
