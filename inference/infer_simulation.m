clc;
clearvars;
close all;

%set random seed
rng(uint32(335747226), 'twister');

%load data and add shading correction
block_data = AbsBlock_July16_30deg();
block_data.addDefaultShadingCorrector();

%get random permutation for each image
index = randperm(block_data.n_sample);
n_train = round(block_data.n_sample/2);
n_artist = block_data.n_sample - n_train;
%get the segmentation image
segmentation = block_data.getSegmentation();
%get the number of segmented images
n_pixel = sum(sum(segmentation));

meanvar_index = index(1:(n_train-1));
test_index = index(n_train-1);
artist_index = index((n_train+1):end);

aRTist = mean(block_data.loadImageStack(artist_index),3);

fig = figure;
imagesc(aRTist);
colorbar;
fig.CurrentAxes.XTick = [];
fig.CurrentAxes.YTick = [];

%get the training images
training_stack = block_data.loadImageStack(meanvar_index);
%segment the image
training_stack = reshape(training_stack,block_data.area,n_train-1);
training_stack = training_stack(reshape(segmentation,[],1),:);
%get the segmented mean and variance greyvalue
training_mean = mean(training_stack,2);
training_var = var(training_stack,[],2);

%train glm using the training set mean and variance
model = GlmGamma(1,IdentityLink());
model.setShapeParameter((n_train-2)/2);
model.train(training_mean,training_var);

%predict variance given aRTist
var_predict = reshape(model.predict(reshape(aRTist,[],1)),block_data.height, block_data.width);

%get the test images
test_0 = block_data.loadImageStack(test_index);

n_intensity = 20;
intensity_array = linspace(1E3, 3E3, n_intensity);
power_array = zeros(1,n_intensity);

got_half = false;

for i_intensity = 1:n_intensity
    
    defect_simulator = DefectSimulator(test_0);
    defect_simulator.addSquareDefectGrid([8;8],[76;76],intensity_array(i_intensity));
    defect_simulator.addPlane( (1E4/(sqrt(2)*1000))*[1;1], 0);
    test = defect_simulator.image;

    z_image = (test - aRTist)./sqrt(var_predict);
    z_image(~segmentation) = nan;

    convolution = EmpericalConvolution(z_image,20, 20, [200,200]);
    convolution.estimateNull(1000);
    convolution.setMask(segmentation);
    convolution.doTest();
    
    sig_0 = defect_simulator.sig_image;
    sig_0(~segmentation) = 0;
    power_array(i_intensity) = sum(sum(convolution.sig_image & sig_0)) / sum(sum(sig_0));
    
    if ~got_half
        if power_array(i_intensity) > 0.5
            
            fig = figure;
            imagesc(-log10(convolution.p_image));
            colorbar;
            hold on;
            colorbar;
            fig.CurrentAxes.XTick = [];
            fig.CurrentAxes.YTick = [];

            fig = figure;
            imagesc(test);
            colorbar;
            hold on;
            colorbar;
            fig.CurrentAxes.XTick = [];
            fig.CurrentAxes.YTick = [];

            fig = figure;
            imagesc(test);
            colorbar;
            hold on;
            colorbar;
            [critical_y, critical_x] = find(convolution.sig_image);
            scatter(critical_x, critical_y,'r.');
            fig.CurrentAxes.XTick = [];
            fig.CurrentAxes.YTick = [];

            fig = figure;
            imagesc(convolution.mean_null);
            colorbar;
            fig.CurrentAxes.XTick = [];
            fig.CurrentAxes.YTick = [];

            fig = figure;
            imagesc(sqrt(convolution.var_null));
            colorbar;
            fig.CurrentAxes.XTick = [];
            fig.CurrentAxes.YTick = [];
            
            got_half = true;
        end
    end
    
end


figure;
plot(intensity_array, power_array);
xlabel('defect intensity');
ylabel('statistical power');
