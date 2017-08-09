%SIGNIFICANT FDR
%Given an array of p values, return boolean array, true if that p value is significant using FDR
%PARAMETERS:
    %p_array: column vector of p values, may contain nan and these are ignored
    %size_test: the size of the test
%RETURN:
    %sig_array: vector of booleans, true if that p value is significant, false otherwise or if nan
function sig_array = significantFDR(p_array, size_test)

    %convert p_array to a column vector
    if isrow(p_array)
        p_array = p_array';
    end
    
    %get boolean vector, true if nan
    nan_index = isnan(p_array);
    %get number of nan nan p values
    m = sum(~nan_index);
    
    %declare array of booleans, all false
    sig_array = zeros(numel(p_array),1);
    
    %remove nan
    p_array(nan_index) = [];
    
    %sort the p_array in accending order
    %p_ordered is p_array sorted
    %p_ordered_index contains indices of the values in p_ordered in relation to p_array
    [p_ordered, p_ordered_index] = sort(p_array);
    
    %find the index of p_ordered which is most significant using the FDR algorithm
    p_critical_index = find( p_ordered <= size_test*(1:m)'/m, 1, 'last');
    
    %if there are p values which are critical
    if ~isempty(p_critical_index)
        %set everything in p_array to be false
        %they will be set to true for significant p values
        p_array = zeros(numel(p_array),1);
        
        %using the entries indiciated by p_ordered_index from element 1 to p_critical_index
        %set these elements in sig_array to be true
        p_array(p_ordered_index(1:p_critical_index)) = true;
        
        %put p_array in non nan entries of sig_array
        sig_array(~nan_index) = p_array;
    end

end

