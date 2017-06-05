%SIGNIFICANT FDR
%Given an array of p values, return boolean array, true if that p value is significant using FDR
%PARAMETERS:
    %p_array: array of p values
    %size_test: the size of the test
%RETURN:
    %sig_array: vector of booleans, true if that p value is significant
function sig_array = significantFDR(p_array, size_test)

    %p_array = p_array(isfinite(p_array));

    %convert p_array to a row vector
    if iscolumn(p_array)
        p_array = p_array';
    end
    
    %get the number of p values
    m = numel(p_array);
    
    %sort the p_array, p_ordered is p_array sorted
    %p_ordered_index contains indices of the values in p_ordered
    [p_ordered, p_ordered_index] = sort(p_array);
    
    %find the index of p_ordered which is significant using the FDR algorithm
    p_critical_index = find( p_ordered <= size_test*(1:m)/m, 1, 'last');
    
    %declare array of booleans, all false
    sig_array = zeros(m,1);
    
    %if there are p values which are critical
    if ~isempty(p_critical_index)
        %using the values of p_ordered_index from element 1 to critical index as indices
        %set these elements in sig_array to be true
        sig_array(p_ordered_index(1:p_critical_index)) = true;
    end

end

