%BGW AUTOCORRELATION SCRIPT
%

clc;
clearvars;
close all;

%number of lags to investigate
n_lag = 1000;

%load the data
block_data = BlockData_140316('data/140316');

%plot the marginal mean greyvalue
%for the black, grey and white images
for i_image = 1:3
    
    %load the b/g/w image
    switch i_image
        case 1
            image = block_data.loadBlack(1);
        case 2
            image = block_data.loadGrey(1);
        case 3
            image = block_data.loadWhite(1);
    end
    
    %x is the mean over the rows
    x = mean(image,1);
    %y is the mean over the columns
    y = mean(image,2);
    
    %plot the marginal mean greyvalues
    figure;
    %plot the marginal greyvalues in the direction of the x axis
    subplot(2,1,1);
    plot(x);
    ylabel('Greyvalue');
    xlabel('Distance in the x axis (pixel)');
    %plot the marginal greyvalues in the direction of they axis
    subplot(2,1,2);
    plot(y);
    ylabel('Greyvalue');
    xlabel('Distance in the y axis (pixel)');
    
end

%declare a vector of autocorrelations, one element for each lag (1 to n_lag+1)
%one vector for the x-direction, y-direction
autocorr_x = zeros(block_data.height,n_lag+1);
autocorr_y = zeros(block_data.width,n_lag+1);

%declare an array of strings, naming each colour
colour_name = {'black','grey','white'};

%for no shading correction, then with shading correction
for i_shading = 1:2
    
    %reset the data
    block_data = BlockData_140316('data/140316');
    
    %if want shading correction
    if i_shading == 2
        %declare array of images, reference stack is an array of b/g/w images
        reference_stack = zeros(block_data.height, block_data.width, 2);
        %load mean b/w images
        reference_stack(:,:,1) = block_data.loadBlack(2);
        reference_stack(:,:,2) = block_data.loadWhite(2);
        %instantise shading corrector using provided reference stack
        shading_corrector = ShadingCorrector_polynomial(reference_stack);
        block_data.addManualShadingCorrector(shading_corrector,[2,2,2]);
    end
    
    %remove dead pixels
    block_data.turnOnRemoveDeadPixels();

    %declare an array of images, one element for b/g/w images
    image_array = zeros(block_data.height, block_data.width, 3);
    image_array(:,:,1) = block_data.loadBlack(1);
    image_array(:,:,2) = block_data.loadGrey(1);
    image_array(:,:,3) = block_data.loadWhite(1);

    %for each colour (b/g/w)
    for i_image = 1:3

        figure;

        %get the image from image_array
        image = image_array(:,:,i_image);

        %for each row
        for y = 1:block_data.height
            %estimate the autocorrelation in the x direction
            autocorr_x(y,:) = reshape(autocorr(image(y,:),n_lag),1,[]);
        end

        %for each column
        for x = 1:block_data.width
            %estimate the autocorrelation in the y direction
            autocorr_y(x,:) = reshape(autocorr(image(:,x),n_lag),1,[]);
        end

        %get the mean autocorrelation over rows/columns
        mean_autocorr_x = mean(autocorr_x);
        mean_autocorr_y = mean(autocorr_y);

        %plot heatmap of autocorrelation for each row
        subplot(2,2,1);
        ax = imagesc(autocorr_x,[-1,1]);
        colorbar('southoutside');
        ylabel('y coordinate');
        xlabel('lag');

        %plot heatmap of autocorrelation for each column
        subplot(2,2,2);
        imagesc(autocorr_y,[-1,1]);
        colorbar('southoutside');
        ylabel('x coordinate');
        xlabel('lag');
        
        %plot mean autocorrelation vs lag in the x direction
        subplot(2,2,3);
        bar(mean_autocorr_x);
        xlim([2,n_lag]);
        ylim(quantile(mean_autocorr_x,[0.01,0.99]));
        ylabel('autocorr. in x direction');
        xlabel('lag');

        %plot mean autocorrelation vs lag in the y direction
        subplot(2,2,4);
        bar(mean_autocorr_y);
        xlim([2,n_lag]);
        ylim(quantile(mean_autocorr_y,[0.01,0.99]));
        ylabel('autocorr. in y direction');
        xlabel('lag');
    end
end
