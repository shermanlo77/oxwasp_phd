%SIGNIFICANT FDR
%Given an array of p values, return boolean array, true if that p value is significant using FDR
%PARAMETERS:
    %p_array: column vector of p values, may contain nan and these are ignored
    %size_test: the size of the test
    %want_plot: boolean, true to plot the p values in a curve
%RETURN:
    %sig_array: vector of booleans, true if that p value is significant, false otherwise or if nan
    %new_size: the new size of the test
function [sig_array, new_size] = significantFDR(p_array, size_test, want_plot)

    %the non-FDR controlled size of the test is the size of the test
    new_size = size_test;

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
        
        new_size = p_ordered(p_critical_index);
        
        %set everything in p_array to be false
        %they will be set to true for significant p values
        p_array = zeros(numel(p_array),1);
        
        %using the entries indiciated by p_ordered_index from element 1 to p_critical_index
        %set these elements in sig_array to be true
        p_array(p_ordered_index(1:p_critical_index)) = true;
        
        %put p_array in non nan entries of sig_array
        sig_array(~nan_index) = p_array;
    end
    
    %if request a plot, plot the p values in order
        %x axis: index number
        %y axis: p value
    if ((nargin==3) && want_plot)
        %plot the p values in order
        figure;
        plot(p_ordered);
        hold on;
        %plot the fdr line
        plot([1,m],[size_test/m,size_test]);
        %if the significant level has been adjusted
        if ~isempty(p_critical_index)
            %adjust the x and y limits
            %x axis adjusted at round(p_critical_index*1.1)
            p_index_plot = min([m,round(p_critical_index*1.1)]);
            xlim([0,p_index_plot]);
            ylim([0,p_ordered(p_index_plot)]);
        end
        %label the axis
        xlabel('Test number');
        ylabel('p value');
    end

end

