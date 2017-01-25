%PROCEDURE: PLOT HISTOGRAM HEAT MAP OF THE MEAN VS VARIANCE RELATIONSHIP
    %note: this function is for ALL the mean-variance pairs#
%PARAMETERS:
    %sample_mean: n vector of the sample mean grey values
    %sample_var: n vector of the sample variance grey values
function [ax] = plotHistogramHeatmap(sample_mean,sample_var,nbin,mean_lim,var_lim)

    %check the parameters are of the correct type
    checkParameters(sample_mean,sample_var);
    
    if nargin == 3
        %get the index of sample_mean and sample variance which are notoutliers
        index = removeOutliers(sample_mean) & removeOutliers(sample_var);
    else
        index = (mean_lim(1) < sample_mean) & (sample_mean < mean_lim(2));
        index = index & (var_lim(1) < sample_var) & (sample_var < var_lim(2));
    end
    
    %remove outliers
    sample_mean = sample_mean(index);
    sample_var = sample_var(index);

    %number of marginal bins to bin the data into
    if nargin==2
        nbin = 3000;
    end
    %bin the data
    [N,c] = hist3([sample_var,sample_mean],[nbin,nbin]);
    
    %plot the heatmap
    ax = axes;
    if nargin == 5
        imagesc(mean_lim,var_lim,[0,0]);
        hold on;
    end
    %normalize N so that the colormap is the frequency density
    imagesc(cell2mat(c(2)),cell2mat(c(1)),N/( (c{2}(2)-c{2}(1))*(c{1}(2)-c{1}(1)) ) );
    axis xy; %switch the y axis
    colorbar; %display the colour bar
    xlabel('Sample grey value mean (arb. unit)'); %label the axis
    ylabel('Sample grey value variance {(arb. unit^2)}'); %label the axis
    xlim([min(sample_mean),max(sample_mean)]);
    ylim([min(sample_var),max(sample_var)]);
    
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

    %NEST FUNCTION: REMOVE OUTLIERS
    %Remove outliers in the vector x using Q(1,3) +/- 1.5 IQR
    function index = removeOutliers(x)
        %get q1 and q3
        q = prctile(x,[25,75]);
        q1 = q(1);
        q2 = q(2);
        %work out iqr
        iqr = q2-q1;
        %find the index which point to data which are not outliers
        index =  ( x > (q1 - 1.5*iqr) );
        index = index & ( x < (q2 + 1.5*iqr) );
    end

end
