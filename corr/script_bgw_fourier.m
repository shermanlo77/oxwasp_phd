clc;
clearvars;
close all;

%plot the marginal mean greyvalue
%for the black, grey and white images
for i_shading = 1:4
    
    %load the data
    block_data = BlockData_140316();
    
    %add the i_shading corresponding shading correction when required
    switch i_shading
        case 2
            %declare array of images, reference stack is an array of b/g/w images
            reference_stack = zeros(block_data.height, block_data.width, 2);
            %load mean b/w images
            reference_stack(:,:,1) = block_data.loadBlack(2);
            reference_stack(:,:,2) = block_data.loadWhite(2);
            %instantise shading corrector using provided reference stack
            shading_corrector = ShadingCorrector(reference_stack);
            block_data.addManualShadingCorrector(shading_corrector);
        case 3
            %declare array of images, reference stack is an array of b/g/w images
            reference_stack = zeros(block_data.height, block_data.width, 3);
            %load mean b/w images
            reference_stack(:,:,1) = block_data.loadBlack(2);
            reference_stack(:,:,2) = block_data.loadWhite(2);
            reference_stack(:,:,3) = block_data.loadGrey(2);
            %instantise shading corrector using provided reference stack
            shading_corrector = ShadingCorrector(reference_stack);
            block_data.addManualShadingCorrector(shading_corrector);
        case 4
            %declare array of images, reference stack is an array of b/g/w images
            reference_stack = zeros(block_data.height, block_data.width, 3);
            %load mean b/w images
            reference_stack(:,:,1) = block_data.loadBlack(2);
            reference_stack(:,:,2) = block_data.loadWhite(2);
            reference_stack(:,:,3) = block_data.loadGrey(2);
            %instantise shading corrector using provided reference stack
            shading_corrector = ShadingCorrector_polynomial(reference_stack,block_data.panel_counter,[2,2,2]);
            block_data.addManualShadingCorrector(shading_corrector);
    end
    block_data.turnOnRemoveDeadPixels();
    
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

        figure;

        L = numel(x);
        f_array = (0:(L/2))/L;
        x_fft = fft(x);
        x_fft = x_fft(1:L/2+1);
        P_x = abs(x_fft/L);
        P_x(2:end-1) = 2*P_x(2:end-1);

        subplot(2,1,1);
        plot(f_array,(P_x));
        ylabel('Magnitude (log)');
        xlabel('Frequency in the x axis (pixel^{-1})');

        L = numel(y);
        f_array = (0:(L/2))/L;
        y_fft = fft(x);
        y_fft = y_fft(1:L/2+1);
        P_y = abs(y_fft/L);
        P_y(2:end-1) = 2*P_y(2:end-1);

        subplot(2,1,2);
        plot(f_array,(P_y));
        ylabel('Magnitude (log)');
        xlabel('Frequency in the y axis (pixel^{-1})');


    end
end