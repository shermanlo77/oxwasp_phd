%PROCEDURE: PLOT HISTOGRAM HEAT MAP OF THE MEAN VS VARIANCE RELATIONSHIP
    %note: this function is for ALL the mean-variance pairs#
%PARAMETERS:
    %sample_mean: n vector of the sample mean grey values
    %sample_var: n vector of the sample variance grey values
function [ax] = hist3Heatmap(x, y, n_bins, is_log_scale)

    %check the parameters are of the correct type
    checkParameters(x,y);
    
    %bin the data
    [N,c] = hist3([y,x],n_bins);
    
    ax = axes;
    
    %normalize N so that the colormap is the frequency density
    if is_log_scale
        imagesc(cell2mat(c(2)),cell2mat(c(1)), log10(N/( (c{2}(2)-c{2}(1))*(c{1}(2)-c{1}(1)) ) ) );
    else
        imagesc(cell2mat(c(2)),cell2mat(c(1)), N/( (c{2}(2)-c{2}(1))*(c{1}(2)-c{1}(1)) )  );
    end
    
    axis xy; %switch the y axis
    
    %NESTED FUNCTION: CHECK PARAMETERS
    function checkParameters(sample_mean,sample_var)
        %check if sample_mean is a column vector, if not throw
        if ~iscolumn(sample_mean)
            error('Error in plotHistogramHeatmap(sample_mean,sample_var), sample_mean is not a column vector');
        end
        %check if sample_var is a column vector, if not throw
        if ~iscolumn(sample_var)
            error('Error in plotHistogramHeatmap(sample_mean,sample_var), sample_var is not a column vector');
        end
        %check if sample_mean and sample_var has the same length, else throw
        n1 = numel(sample_mean);
        n2 = numel(sample_var);
        if n1~=n2
            error('Error in plotHistogramHeatmap(sample_mean,sample_var), sample_mean and sample_var are not the same length');
        end
    end

end
