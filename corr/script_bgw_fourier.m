clc;
clearvars;
%close all;

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

    x = x-mean(x);
    y = y-mean(y);
    
    figure;

    L = numel(x);
    f_array = (1:(L/2))/L; %change 1 to zero if using regular fft
    x_fft = fft(x);
%     P_x = abs(x_fft/L);
%     P_x = P_x(1:L/2+1);
%     P_x(2:end-1) = 2*P_x(2:end-1);
%     P_x = periodogram(x);
%     P_x = P_x(1:numel(f_array));
    P_x = fft_leastSquares(x);
    
    subplot(2,1,1);
    plot(f_array,log10(P_x));
    ylabel('Magnitude (log)');
    xlabel('Frequency in the x axis (pixel^{-1})');

    L = numel(y);
    f_array = (1:(L/2))/L; %change 1 to zero if using regular fft
    y_fft = fft(y);
%     P_y = abs(y_fft/L);
%     P_y = P_y(1:L/2+1);
%     P_y(2:end-1) = 2*P_y(2:end-1);
%     P_y = periodogram(y);
%     P_y = P_y(1:numel(f_array));
    P_y = fft_leastSquares(y);

    subplot(2,1,2);
    plot(f_array,log10(P_y));
    ylabel('Magnitude (log)');
    xlabel('Frequency in the y axis (pixel^{-1})');


end