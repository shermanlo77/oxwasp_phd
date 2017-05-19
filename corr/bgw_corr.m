clc;
clearvars;
close all;

block_data = BlockData_140316('data/140316');
block_data.addShadingCorrector(@ShadingCorrector_polynomial,true,[2,2,2]);
block_data.turnOnRemoveDeadPixels();

image_array = zeros(block_data.height, block_data.width, 3);
image_array(:,:,1) = block_data.loadBlack(1);
image_array(:,:,2) = block_data.loadGrey(1);
image_array(:,:,3) = block_data.loadWhite(1);

n_lag = 1000;

autocorr_x = zeros(block_data.height,n_lag+1);
autocorr_y = zeros(block_data.width,n_lag+1);

colour_name = {'black','grey','white'};

for i_image = 1:3
    
    figure;
    
    image = image_array(:,:,i_image);

    for y = 1:block_data.height

        autocorr_x(y,:) = reshape(autocorr(image(y,:),n_lag),1,[]);

    end

    for x = 1:block_data.width

        autocorr_y(x,:) = reshape(autocorr(image(:,x),n_lag),1,[]);

    end

    subplot(2,2,1);
    imagesc(autocorr_x,[-1,1]);
    colorbar('southoutside')
    ylabel('y coordinate');
    xlabel('lag');

    subplot(2,2,2);
    imagesc(autocorr_y,[-1,1]);
    colorbar('southoutside')
    ylabel('x coordinate');
    xlabel('lag');
    
    
    mean_autocorr_x = mean(autocorr_x);
    mean_autocorr_y = mean(autocorr_y);

    subplot(2,2,3);
    bar(mean_autocorr_x);
    xlim([0,n_lag]);
    ylabel('autocorr. in x direction');
    xlabel('lag');
    
    subplot(2,2,4);
    bar(mean_autocorr_y);
    xlim([0,n_lag]);
    ylabel('autocorr. in y direction');
    xlabel('lag');
end
