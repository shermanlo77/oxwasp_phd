%GLM FIT FOR DIFFERENT SAMPLE SIZES
%Fit gamma GLM on the mean and variance relationship. This is done 4
%different times using different sample sizes when calculating the mean and
%variance estimates data. The frequency density is ploted with the fit.

clc;
clearvars;
close all;

%set random seed
rng(uint32(189224219), 'twister');

%instantise an object pointing to the dataset
block_data = AbsBlock_Mar16();

%number of bins for the frequency density plot
nbin = 100;

%array of sample sizes
n_sample_array = [25,50,75,100];

%array of figures
fig_array = cell(1,numel(n_sample_array));
axe_array = cell(1,numel(n_sample_array));

%object for estimating the mean and variance
mean_var_estimator = MeanVarianceEstimator(block_data);

%for each sample size
for i_sample = 1:numel(n_sample_array)
    
    %get the sample size
    n_sample = n_sample_array(i_sample);
    
    %set n_sample of images
    data_index = randperm(block_data.n_sample);
    data_index = data_index(1:n_sample);
    %work out the mean and variance over these n_sample images
    [sample_mean,sample_var] = mean_var_estimator.getMeanVar(data_index);

    %shape parameter is number of (images - 1)/2, this comes from the chi squared distribution
    shape_parameter = (n_sample-1)/2;

    %model the mean and variance using gamma glm
    model = GlmGamma(1,IdentityLink());
    model.setShapeParameter(shape_parameter);
    %train the classifier
    model.train(sample_mean,sample_var);

    %get a range of greyvalues to plot the fit
    x_plot = linspace(min(sample_mean),max(sample_mean),100);
    %get the variance prediction along with the error bars
    [variance_prediction, up_error, down_error] = model.predict(x_plot');

    %plot the frequency density
    fig_array{i_sample} = LatexFigure.main();
    axe_array{i_sample} = hist3Heatmap(sample_mean,sample_var,[nbin,nbin],true);
    hold on;
    %plot the fit/prediction
    plot(x_plot,variance_prediction,'r');
    %plot the error bars
    plot(x_plot,up_error,'r--');
    plot(x_plot,down_error,'r--');
    colorbar;
    xlabel('mean (arb. unit)');
    ylabel('variance (arb. unit)');
end

%rescale the colorbar
%declare an array of maximum frequency density, one for each figure or sample size
max_frequency_density = zeros(1,numel(n_sample_array));
%for each sample size
for i_sample = 1:numel(n_sample_array)
    %get the maximum value in the figure
    max_frequency_density(i_sample) = axe_array{i_sample}.CLim(end);
end
%find the maximum maximum value
max_frequency_density = max(max_frequency_density);
%for each figure, set CLim to the that maximum value
for i_sample = 1:numel(n_sample_array)
    axe_array{i_sample}.CLim(end) = max_frequency_density;
end

%rescale the x and y axis
%x_lim_array and y_lim_array is a collection of XLim and YLim of each axes
x_lim_array = zeros(numel(n_sample_array),2);
y_lim_array = zeros(numel(n_sample_array),2);
%get the colour of the background from the currect colormap
blank_colour = colormap;
blank_colour = blank_colour(1,:);
%for each axes
for i_sample = 1:numel(n_sample_array)
    %get XLim and YLim and save it to x_lim_array and y_lim_array
    x_lim_array(i_sample,:) = axe_array{i_sample}.XLim;
    y_lim_array(i_sample,:) = axe_array{i_sample}.YLim;
end
%set x_lim and y_lim to cover all the collected XLim and YLim
x_lim = [min(x_lim_array(:,1)),max(x_lim_array(:,2))];
y_lim = [min(y_lim_array(:,1)),max(y_lim_array(:,2))];
%for each axes
for i_sample = 1:numel(n_sample_array)
    %set XLim and YLim
    axe_array{i_sample}.XLim = x_lim;
    axe_array{i_sample}.YLim = y_lim;
    %set the colour of the background
    axe_array{i_sample}.set('color',blank_colour);
end

%for each axes
for i_sample = 1:numel(n_sample_array)
    %export the background
    fig_array{i_sample}.InvertHardcopy = 'off';
    %set the background to white (of the figure)
%     fig_array{i_sample}.Color = 'white';
    %export the figure
    saveas(fig_array{i_sample},fullfile('reports','figures','meanVar',strcat('sample_size_',num2str(n_sample_array(i_sample)),'.eps')),'epsc');
end