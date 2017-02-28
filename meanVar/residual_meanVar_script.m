clc;
clearvars;
close all;

n_train = 50;
threshold = BlockData_140316.getThreshold_topHalf();

%location of the data
block_location = '../data/140316';
%shape parameter
shape_parameter = (n_train-1)/2;

%declare array of block images (3 of them)
block_array = cell(3,1);
%1st one has no shading correction
block_array{1} = BlockData_140316(block_location);
%2nd one uses b/w for shading correction
block_array{2} = BlockData_140316(block_location);
block_array{2}.addShadingCorrector(@ShadingCorrector,false);
%3rd one uses b/g/w for shading correction
block_array{3} = BlockData_140316(block_location);
block_array{3}.addShadingCorrector(@ShadingCorrector,true);

%instantiate identity model 
glm_model = MeanVar_GLM_identity(shape_parameter,1);

%declare array of residuals
    %dim_1: for each block
    %dim_2: residual, standard residuals
residual_array = cell(numel(block_array),2);

%for each block
for i_block = 1:numel(block_array)
    
    %get the block
    data = block_array{i_block};
    
    %get random index of the training and test data
    index_suffle = randperm(data.n_sample);
    training_index = index_suffle(1:n_train);
    test_index = index_suffle((n_train+1):data.n_sample);

    %get variance mean data of the training set
    [sample_mean,sample_var] = data.getSampleMeanVar_topHalf(training_index);
    %segment the mean var data
    sample_mean(threshold) = [];
    sample_var(threshold) = [];

    %train the classifier
    glm_model.train(sample_mean,sample_var);

    %get the variance mean data of the test set
    [sample_mean,sample_var] = data.getSampleMeanVar_topHalf(test_index);
    %segment the mean var data
    sample_mean(threshold) = [];
    sample_var(threshold) = [];
    
    %get the residual and the standardised residual
    residual = (sample_var - glm_model.predict(sample_mean));
    standard_residual = residual ./ sqrt(glm_model.getVariance(sample_mean));
    
    %save the greyvvalue and residual to residual_array
    residual_array{i_block,1} = [sample_mean, residual];
    residual_array{i_block,2} = [sample_mean, standard_residual];
    
end

%declare array of axes
    %dim_1: for each block
    %dim_2: residual, standard residuals
ax_array = cell(numel(block_array),2);

%for non-standard and standard residuals
for i_ax = 1:2
    
    %get the first set of residuals
    residual = residual_array{1,i_ax};
    
    %work out the outlier boundary for the greyvalue and residuals
    q = quantile(residual,[0.25,0.75]);
    iqr = q(2,:) - q(1,:);
    %x_lim_max and y_lim_max is the biggest axes limit
    x_lim_max = [q(1,1) - 1.5*iqr(1), q(2,1) + 1.5*iqr(1)];
    y_lim_max = [q(1,2) - 1.5*iqr(2), q(2,2) + 1.5*iqr(2)];
    
    %for the remaining set of residuals
    for i_block = 2:numel(block_array)
        
        %get the resdual
        residual = residual_array{1,i_ax};
        
        %work out the outlier boundary for the greyvalue and residuals
        q = quantile(residual,[0.25,0.75]);
        iqr = q(2,:) - q(1,:);
        x_lim = [q(1,1) - 1.5*iqr(1), q(2,1) + 1.5*iqr(1)];
        y_lim = [q(1,2) - 1.5*iqr(2), q(2,2) + 1.5*iqr(2)];
        
        %update x_lim_max
        if x_lim(1) < x_lim_max(1)
            x_lim_max(1) = x_lim(1);
        end
        if x_lim(2) > x_lim_max(2)
            x_lim_max(2) = x_lim(2);
        end
        
        %update y_lim_max
        if y_lim(1) < y_lim_max(1)
            y_lim_max(1) = y_lim(1);
        end
        if y_lim(2) > y_lim_max(2)
            y_lim_max(2) = y_lim(2);
        end
        
    end
    
    %for each block
    for i_block = 1:numel(block_array)
        %get the residual
        residual = residual_array{i_block,i_ax};
        %plot the frequency density of the residuals
        figure;
        ax_array{i_block,i_ax} = plotHistogramHeatmap(residual(:,1),residual(:,2),50,x_lim_max,y_lim_max);
        %set the axes limit
        ax_array{i_block,i_ax}.XLim = x_lim_max;
        ax_array{i_block,i_ax}.YLim = y_lim_max;
        %label the y axes
        if i_ax == 1
            ylabel(ax_array{i_block,i_ax},'Residual {(arb. unit^2)}');
        else
            ylabel(ax_array{i_block,i_ax},'Standard residual');
        end
    end
end

%get the colour of the background
blank_colour = colormap;
blank_colour = blank_colour(1,:);

%for non-standard and standard residuals
for i_ax = 1:2
    %get the first c_limit (limit from the colour in the density plot)
    %represent the maximum c_limit
    c_lim_max = ax_array{1,i_ax}.CLim(2);
    %for the remaining graphs
    for i_block = 2:numel(block_array)
        %update c_lim_max
        if c_lim_max < ax_array{i_block,i_ax}.CLim(2)
            c_lim_max = ax_array{i_block,i_ax}.CLim(2);
        end
    end
    %for all graphs
    for i_block = 1:numel(block_array)
        %get CLim and the background colour
        ax_array{i_block,i_ax}.CLim(2) = c_lim_max;
        ax_array{i_block,i_ax}.set('color',blank_colour);
    end
end