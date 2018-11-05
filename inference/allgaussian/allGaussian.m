%SCRIPT: ALL GAUSSIAN
%Filters a Gaussian image
%Shows the empirical null mean and std image
%Plots the qq plot of the post filter greyvalues

clc;
clearvars;
close all;

randStream = RandStream('mt19937ar','Seed',uint32(3499211588)); %instantise a rng

imageSize = 256;
radius = 20; %radius of kernel

image = randStream.randn(256,256); %create gaussian image
filter = EmpiricalNullFilter(radius); %filter it
filter.setNInitial(1);
filter.filter(image);

%get the empirical null and the filtered image
imageFiltered = filter.getFilteredImage();
nullMean = filter.getNullMean();
nullStd = filter.getNullStd();

%qq plot
figure;
qqplot(reshape(imageFiltered,[],1));

%empirical null plot
figure;
imagesc(nullMean);
colorbar;
figure;
imagesc(nullStd);
colorbar;