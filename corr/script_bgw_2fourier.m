clc;
clearvars;
close all;

%plot the marginal mean greyvalue
%for the black, grey and white images
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
    
    figure;
    for i_image = 1:2

        %load the b/g/w image
        switch i_image
            case 1
                image = bgw_data.reference_scan_array(1).loadImage(1);
            case 2
                image = bgw_data.reference_scan_array(3).loadImage(1);
        end

        ax = imagesc_truncate(abs(fftshift(fft2(image))));
        subplot(1,2,i_image,ax);

    end
end