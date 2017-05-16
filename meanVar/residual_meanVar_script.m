%RESIDUAL MEAN VARIANCE SCRIPT
%For 3 different types of shading correction (no shading correction, bw
%shading correction, bgw shading correction), spilt the 100 images into two
%equally sized sets: training set, test set. Work out the sample mean and
%sample variance within pixel grey value for both the training and test
%set. Train GLM on training set, predict variance given mean using test
%set.
%Plot error / std of glm response vs mean (residual = error)

clc;
clearvars;
close all;

rng(uint32(373499714), 'twister');

n_train = 50;
threshold = BlockData_140316.getThreshold_topHalf();

%location of the data
block_location = 'data/140316';
%shape parameter
shape_parameter = (n_train-1)/2;

%declare array of block images (3 of them)
block_array = cell(3,1);
%1st one has no shading correction
block_array{1} = BlockData_140316(block_location);
%2nd one uses b/w for shading correction
block_array{2} = BlockData_140316(block_location);
block_array{2}.addShadingCorrector(@ShadingCorrector,false);
block_array{2}.turnOnRemoveDeadPixels();
block_array{2}.turnOnSetExtremeToNan();
%3rd one uses b/g/w for shading correction
block_array{3} = BlockData_140316(block_location);
block_array{3}.addShadingCorrector(@ShadingCorrector,true);
block_array{3}.turnOnRemoveDeadPixels();
block_array{3}.turnOnSetExtremeToNan();

%instantiate identity model 
glm_model = MeanVar_GLM_identity(shape_parameter,1);

%declare array of residuals
    %dim_1: for each block
residual_array = cell(numel(block_array),1);

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
    residual_array{i_block} = [sample_mean, standard_residual];
    
end

%declare array of figures and exes
    %dim_1: for each block
    %dim_2: axes, figure
ax_array = cell(numel(block_array),2);
    
%get the first set of residuals
residual = residual_array{1};

%work out the outlier boundary for the greyvalue and residuals
q = quantile(residual,[0.25,0.75]);
iqr = q(2,:) - q(1,:);
%x_lim_max and y_lim_max is the biggest axes limit
x_lim_max = [q(1,1) - 1.5*iqr(1), q(2,1) + 1.5*iqr(1)];
y_lim_max = [q(1,2) - 1.5*iqr(2), q(2,2) + 1.5*iqr(2)];

%for the remaining set of residuals
for i_block = 2:numel(block_array)

    %get the resdual
    residual = residual_array{1};

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
    residual = residual_array{i_block};
    %plot the frequency density of the residuals
    ax_array{i_block,2} = figure;
    ax_array{i_block,1} = plotHistogramHeatmap(residual(:,1),residual(:,2),50,x_lim_max,y_lim_max);
    %set the axes limit
    ax_array{i_block,1}.XLim = x_lim_max;
    ax_array{i_block,1}.YLim = y_lim_max;
    ylabel(ax_array{i_block,1},'Standard residual');
end

%get the colour of the background
blank_colour = colormap;
blank_colour = blank_colour(1,:);

%get the first c_limit (limit from the colour in the density plot)
%represent the maximum c_limit
c_lim_max = ax_array{1,1}.CLim(2);
%for the remaining graphs
for i_block = 2:numel(block_array)
    %update c_lim_max
    if c_lim_max < ax_array{i_block,1}.CLim(2)
        c_lim_max = ax_array{i_block,1}.CLim(2);
    end
end
%name for different types of shading correction
name_array = {'no_shad','bw','bgw'};
%for all graphs
for i_block = 1:numel(block_array)
    %get CLim and the background colour
    ax_array{i_block,1}.CLim(2) = c_lim_max;
    ax_array{i_block,1}.set('color',blank_colour);
    %export the background
    ax_array{i_block,2}.InvertHardcopy = 'off';
    %set the background to white (of the figure)
    ax_array{i_block,2}.Color = 'white';
    %save figure
    saveas(ax_array{i_block,2},strcat('reports/figures/meanVar/residual_',name_array{i_block},'.png'),'png');
end