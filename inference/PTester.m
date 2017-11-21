%P VALUE TESTER
%Does multiple hypothesis tests on a given image of p values
%Multiple testing corrected by controlling the FDR
%See:
    %Benjamini, Y. and Hochberg, Y. 1995
    %Controlling the false discovery rate: a practical and powerful approach to multiple testing
    %Journal of the royal statistical society
classdef PTester < handle
    
    %MEMBER VARIABLES
    properties (SetAccess = private)
        p_image; %2d array of p values
        sig_image; %2d boolean array, true of significant pixel
        size; %size of test
        size_corrected; %size of test corrected for multiple testing
        n_test; %number of tests in the image
    end
    
    %METHODS
    methods (Access = public)
       
        %CONSTRUCTOR
        %PARAMETERS:
            %p_image: 2d array of p values
            %size: size of the test
        function this = PTester(p_image, size)
            %assign member variables
            this.p_image = p_image;
            this.sig_image = p_image;
            this.sig_image(:) = false;
            this.size = size;
            
            %get the number of non_nan values in z_image
            nan_index = isnan(reshape(p_image,[],1));
            this.n_test = sum(~nan_index);
        end
        
        %METHOD: DO TEST
        %Do hypothesis test using the given p values, controlling the FDR
        %Assign the member variables sig_image and size_corrected
        function doTest(this)

            %put the p values in a column vector
            p_array = reshape(this.p_image,[],1);
            %sort the p_array in accending order
            %p_ordered is p_array sorted
            %p_ordered_index contains indices of the values in p_ordered in relation to p_array
            [p_ordered, p_ordered_index] = sort(p_array);

            %find the index of p_ordered which is most significant using the FDR algorithm
            p_critical_index = find( p_ordered(~isnan(p_ordered)) <= this.size*(1:this.n_test)'/this.n_test, 1, 'last');

            %if there are p values which are significant
            if ~isempty(p_critical_index)

                %correct the size of the test using that p value
                this.size_corrected = p_ordered(p_critical_index);

                %set everything in p_array to be false
                %they will be set to true for significant p values
                p_array = zeros(numel(p_array),1);

                %using the entries indiciated by p_ordered_index from element 1 to p_critical_index
                %set these elements in sig_array to be true
                p_array(p_ordered_index(1:p_critical_index)) = true;

                %put p_array in non nan entries of sig_array
                this.sig_image(:) = p_array;
            else
                %correct the size of the test is the Bonferroni correction
                this.size_corrected = this.size / this.n_test;
            end
            
        end
        
    end
    
    
end

