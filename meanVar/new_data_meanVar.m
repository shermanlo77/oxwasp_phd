clc;
clearvars;
close all;

block_data = AbsBlock_Sep16_120deg();
image_stack = block_data.loadImageStack();
mean_stack = mean(image_stack,3);
var_stack = var(image_stack,[],3);

mean_stack = reshape(mean_stack,[],1);
var_stack = reshape(var_stack,[],1);

figure;
hist3Heatmap(mean_stack, var_stack, [100,100], true);
colorbar;
xlabel('mean (arb. unit)');
ylabel('variance (arb. unit^2)');