clc;
clearvars;
close all;

%plot the marginal mean greyvalue
%for the black, grey and white images
for i_shading = 1:4
    
    %reset the data
    block_data = BlockData_140316();
    
    %add the i_shading corresponding shading correction when required
    switch i_shading
        case 2
            block_data.addShadingCorrector(ShadingCorrector(),[2,2]);
        case 3
            block_data.addShadingCorrector(ShadingCorrector(),[2,2,2]);
        case 4
            block_data.addShadingCorrector(ShadingCorrector_polynomial([2,2,2]),[2,2,2]);
    end
    
    %remove dead pixels
    block_data.turnOnRemoveDeadPixels();
    
    figure;
    
    for i_image = 1:2

        %load the b/g/w image
        switch i_image
            case 1
                image = block_data.loadBlack(1);
            case 2
                image = block_data.loadWhite(1);
        end

        %x is the mean over the rows
        x = mean(image,1);
        %y is the mean over the columns
        y = mean(image,2);

        x = x-mean(x);
        y = y-mean(y);

        L = numel(x);
        f_array = (0:(L/2))/L;
        x_fft = fft(x);
        x_fft = x_fft(1:L/2+1);
        P_x = abs(x_fft/L);
        P_x(2:end-1) = 2*P_x(2:end-1);

        subplot(2,2,1+(i_image-1)*2);
        plot(f_array,(P_x));
        xlim([-0.01,0.5]);
        ylabel('Amplitude (greyvalue)');
        xlabel('Frequency in the x axis (pixel^{-1})');

        L = numel(y);
        f_array = (0:(L/2))/L;
        y_fft = fft(x);
        y_fft = y_fft(1:L/2+1);
        P_y = abs(y_fft/L);
        P_y(2:end-1) = 2*P_y(2:end-1);

        subplot(2,2,2+(i_image-1)*2);
        plot(f_array,(P_y));
        xlim([-0.01,0.5]);
        ylabel('Amplitude (greyvalue)');
        xlabel('Frequency in the y axis (pixel^{-1})');


    end
end