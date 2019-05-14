%TEST REMOVE DEAD PIXELS FUNCTION
%Using the dataset cameraman.tif, randomly assign nan with probability 0.9
%and then remove the dead pixels.
%PLOTS:
    %orginial image
    %image with nan
    %image with dead pixels removed

clc;
clearvars;
close all;

%set random seed
rng(uint32(112434310), 'twister');

%get the image
slice = double(imread('cameraman.tif'));

%plot the image
figure;
imagesc(slice);
colormap gray;

%get the size of the image
[dim_1,dim_2] = size(slice);
area = dim_1 * dim_2;

%probability of assigning nan
p_nan = 0.9;
%simulate the positions of nan
nan_index = rand(dim_1,dim_2) < p_nan;
%get the x and y coordinate of the nan
[nan_y,nan_x] = find(nan_index);
%assign nan to the allocated pixels
slice(nan_index) = nan;

%plot image with nan
figure;
imagesc(slice);
colormap gray;

%plot image with nan removed using the function removeDeadPixels
figure;
imagesc(removeDeadPixels(slice));
colormap gray;
