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

        ax = imagesc_truncate(abs(fftshift(fft2(image))));
        subplot(1,2,i_image,ax);

    end
end