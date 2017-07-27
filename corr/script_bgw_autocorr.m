%BGW AUTOCORRELATION SCRIPT
%Plots the x,y time series and autocorrelation from the black,white,grey images
%Figures:
    %1-2: b/w shading uncorrected x,y time series
    %3-4: b/w shading uncorrected autocorrelation
    %5-6: b/w BW shading corrected x,y time series
    %7-8: b/w BW shading corrected autocorrelation
    %9-10: b/w BGW shading corrected x,y time series
    %11-12: b/w BGW shading corrected autocorrelation
    %13-14: b/w BGW polynomial shading corrected x,y time series
    %15-16: b/w BGW polynomail shading uncorrected autocorrelation

clc;
clearvars;
close all;

%number of lags to investigate
n_lag = 1000;

%load the data
bgw_data = Bgw_Mar16();

%declare a vector of autocorrelations, one element for each lag (1 to n_lag+1)
%one vector for the x-direction, y-direction
autocorr_x = zeros(bgw_data.height,n_lag+1);
autocorr_y = zeros(bgw_data.width,n_lag+1);

%declare an array of strings, naming each colour
colour_name = {'black','grey','white'};

%for no shading correction, then with 3 differenty types of shading correction
for i_shading = 1:4
    
    %reset the data
    bgw_data = Bgw_Mar16();
    
    %add the i_shading corresponding shading correction when required
    switch i_shading
        case 2
            bgw_data.addDefaultShadingCorrector();
        case 3
            bgw_data.addShadingCorrector(ShadingCorrector(),[1,2,3]);
        case 4
            bgw_data.addShadingCorrector(ShadingCorrector_polynomial([2,2,2]),[1,2,3]);
    end
    
    %remove dead pixels
    bgw_data.turnOnRemoveDeadPixels();
    
    %declare an array of images, one element for b/g/w images
    image_array = zeros(bgw_data.height, bgw_data.width, 3);
    image_array(:,:,1) = bgw_data.reference_scan_array(1).loadImage(1);
    image_array(:,:,2) = bgw_data.reference_scan_array(2).loadImage(1);
    image_array(:,:,3) = bgw_data.reference_scan_array(3).loadImage(1);
    
    %plot the marginal mean greyvalue
    %for the black, grey and white images
    for i_image = [1,3]

        %load the b/g/w image
        %get the image from image_array
        image = image_array(:,:,i_image);

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

    %for each colour (b/w)
    for i_image = [1,3]

        figure;

        %get the image from image_array
        image = image_array(:,:,i_image);

        %for each row
        for y = 1:bgw_data.height
            %estimate the autocorrelation in the x direction
            autocorr_x(y,:) = reshape(autocorr(image(y,:),n_lag),1,[]);
        end

        %for each column
        for x = 1:bgw_data.width
            %estimate the autocorrelation in the y direction
            autocorr_y(x,:) = reshape(autocorr(image(:,x),n_lag),1,[]);
        end

        %get the mean autocorrelation over rows/columns
        mean_autocorr_x = mean(autocorr_x);
        mean_autocorr_y = mean(autocorr_y);
        
        imagesc_range = 3*std([mean_autocorr_x,mean_autocorr_y]);
        if imagesc_range > 1
            imagesc_range = 1;
        end
        
        %plot heatmap of autocorrelation for each row
        subplot(2,2,1);
        ax = imagesc(autocorr_x,[-imagesc_range,imagesc_range]);
        colorbar('southoutside');
        ylabel('y coordinate');
        xlabel('lag');

        %plot heatmap of autocorrelation for each column
        subplot(2,2,2);
        imagesc(autocorr_y,[-imagesc_range,imagesc_range]);
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
