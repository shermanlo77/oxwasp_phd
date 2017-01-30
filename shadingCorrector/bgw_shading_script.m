%BLACK GREY WHITE SHADING CORRECTION SCRIPT
%Spilts the b/g/w images into training and test set. The training set is
%used to calibrate the shading correction (the mean of the training set is
%used, they may be filtered as well). Then each test image is shading
%corrected, the between/within pixel variance is recorded. This is repeated
%20 times.
%
%The mean b/g/w shading corrected is ploted, along with the between/within
%pixel variance.
%
%Shading correction includes: no shading correction, b/w shading correction,
%b/g/w/ shading correction, 2nd order panel wise polynomial

clc;
clearvars;
close all;

%set random seed
rng(uint32(227482200), 'twister');

%load the data
block_data = BlockData_140316('../data/140316');

%repeat the experiment 20 times
n_repeat = 20;

%for the 4 different types of shading correction
for i = 1:4
    
    %do ANOVA analysis on the shading corrected b/g/w images
    %parameters of shadingCorrection_ANOVA
        %data object
        %number of images in the training set
        %function handle for the shading corrector
        %boolean: include grey images
        %paramters for shading correction
        %number of times to repeat the experiment
    switch i
        case 1
            std_array = shadingCorrection_ANOVA(block_data, 10, @ShadingCorrector_null, 0, nan, n_repeat);
        case 2
            std_array = shadingCorrection_ANOVA(block_data, 10, @ShadingCorrector, 0, nan, n_repeat);
        case 3
            std_array = shadingCorrection_ANOVA(block_data, 10, @ShadingCorrector, 1, nan, n_repeat);
        case 4
            std_array = shadingCorrection_ANOVA(block_data, 10, @ShadingCorrector_polynomial, 1, [2,2,2], n_repeat);
    end
    
end
