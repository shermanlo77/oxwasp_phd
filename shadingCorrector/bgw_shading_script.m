clc;
clearvars;
close all;

block_data = BlockData_140316('../data/140316');

n_repeat = 10;

for i = 1:4
    
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
