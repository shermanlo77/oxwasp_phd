%CLASS: BOXPLOTS
%Custom box plot class for plotting multiple box plots
%Uses quantile box plot instead of the standard one
classdef BoxplotsQuantile < Boxplots
  
  methods (Access = public)
    
    function this = BoxplotsQuantile(X)
      this@Boxplots(X);
    end
    
  end
  
  methods (Access = protected)
    
    function boxplot = getBoxplot(this, X)
      boxplot = BoxplotQuantile(X);
    end
    
  end
  
end

