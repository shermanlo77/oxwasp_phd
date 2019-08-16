%MIT License
%Copyright (c) 2019 Sherman Lo

%P VALUE TESTER
%Does multiple hypothesis tests on a given image of p values
%Multiple testing corrected by controlling the FDR
%See:
  %Benjamini, Y. and Hochberg, Y. 1995
  %Controlling the false discovery rate: a practical and powerful approach to multiple testing
  %Journal of the royal statistical society
%How to use:
  %pass p values (can be 2D array) into the constructor along with the fdr threshold
  %call method doTest
  %get the positive results in the member variable positiveImage
classdef PTester < handle
  
  properties (SetAccess = private)
    pImage; %2d array of p values
    positiveImage; %2d boolean array, true of positive pixel
    fdr; %user threshold for fdr of test
    size; %significant level of the test, obtained from BH procedure
    nTest; %number of tests in the image (non-nan)
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    %PARAMETERS:
      %pImage: 2d array of p values
      %fdr: fdr of the test
    function this = PTester(pImage, fdr)
      %assign member variables
      this.pImage = pImage;
      this.positiveImage = false(size(pImage));
      this.fdr = fdr;
      %get the number of non_nan values in z_image
      this.nTest = sum(sum(~isnan(pImage)));
    end
    
    %METHOD: DO TEST
      %Do hypothesis test using the given p values, controlling the FDR
      %Assign the member variables positiveImage and size
    function doTest(this)
      
      %put the p values in a column vector
      pArray = reshape(this.pImage,[],1);
      %remove nan
      pArray(isnan(pArray)) = [];
      %sort the p_array in accending order
        %pOrdered is pArray sorted
        %pOrderedIndex contains indices of the values in pOrdered in relation to pArray
      [pOrdered, pOrderedIndex] = sort(pArray);
      
      %find the index of pOrdered which is most significant using the FDR algorithm
      pCriticalIndex = find( pOrdered <= this.fdr*(1:this.nTest)'/this.nTest, true, 'last');
      
      %if there are p values which are significant
      if ~isempty(pCriticalIndex)
        
        %set the size of the test using that p value
        this.size = pOrdered(pCriticalIndex);
        
        %set everything in pArray to be false
        %they will be set to true for significant p values
        pArray = false(numel(pArray),1);
        
        %using the entries indiciated by pOrderedIndex from element 1 to pCriticalIndex
        %set these elements in positiveImage to be true
        pArray(pOrderedIndex(1:pCriticalIndex)) = true;
        
        %put pArray in non nan entries of positiveImage
        this.positiveImage(~isnan(this.pImage)) = pArray;
        
      else
        %correct the fdr of the test is the Bonferroni correction
        this.size = this.fdr / this.nTest;
        
      end
      
    end
    
  end
  
end

