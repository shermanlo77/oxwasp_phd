%CLASS: Custom class for boxplot
%Whiskers are the symmetric 99.3% quantiles
%The probability was chosen to minic the (q1,q3) +/- 1.5*iqr for all normal data
classdef BoxplotQuantile < Boxplot
  
  %MEMBER VARIABLES
  properties (SetAccess = private)
      alpha = 0.993; %the whiskers capture this amount of the data
      
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    function this = BoxplotQuantile(X)
      %call superclass constructor
      this@Boxplot(X);
      this.setWhiskerCap(true);
    end
    
  end
  
  methods (Access = protected)
    
    %METHOD: GET WHISKER
    %Get the whiskers, that is the min and max of non-outlier data
    function getWhisker(this)
      p = (1-this.alpha)/2;
      this.whisker = quantile(this.X,[p,1-p]);
    end
    
    %METHOD: GET OUTLIER
    %Set which data are outliers or not, save the boolean in the member variable outlier_index
    function getOutlier(this)
      this.outlier_index = false(numel(this.X), 1);
    end
    
  end
  
end

