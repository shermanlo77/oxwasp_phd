%FIT KERNEL SCRIPT
%Regress the mean and variance relationship using kernel regression using different scale parameters
%The frequency density is plotted with the fit

clc;
clearvars;
close all;

%number of bins for the frequency density plot
nbin = 100;

%instantise an object pointing to the dataset
block_data = AbsBlock_Mar16();

%get variance mean data of the top half of the scans (images 1 to 100)
mean_var_estimator = MeanVarianceEstimator(block_data);
[sample_mean,sample_var] = mean_var_estimator.getMeanVar(1:block_data.n_sample);

%get an array of x to evaluate the kernel density at
x_plot = linspace(min(sample_mean),max(sample_mean),500);

%for each k
for k = [1E0, 1E3]
    
    %model the mean and variance using kNN
    model = KernelRegression(EpanechnikovKernel(), k);
    %train the classifier
    model.train(sample_mean,sample_var);

    %get the variance prediction
    variance_prediction = model.predict(x_plot);

    %plot the frequency density
    fig = LatexFigure.main();
    hist3Heatmap(sample_mean,sample_var,[nbin,nbin],true);
    hold on;
    %plot the fit/prediction
    plot(x_plot,variance_prediction,'r');
    %put in the colour bar
    colorbar;
    %label the axis
    xlabel('mean (arb. unit)');
    ylabel('variance (arb. unit)');
    
    %save the figure
    saveas(fig,fullfile('reports','figures','meanVar',strcat('kernel',num2str(log10(k)),'.eps')),'epsc');
end