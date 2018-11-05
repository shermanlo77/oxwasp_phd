clc;
clearvars;
close all;

randStream = RandStream('mt19937ar','Seed',uint32(3499211588)); %instantise a rng

imageSize = 256;
radius = 100;

image = randn(256,256);
filter = EmpiricalNullFilter(radius);
filter.setNInitial(20);
tic;
filter.filter(image);
toc;

imageFiltered = filter.getFilteredImage();
nullMean = filter.getNullMean();
nullStd = filter.getNullStd();

figure;
qqplot(reshape(image,[],1));
figure;
qqplot(reshape(imageFiltered,[],1));

figure;
histogram_custom(reshape(image,[],1));
hold on;
histogram_custom(reshape(imageFiltered,[],1));
hold off;

figure;
imagesc(image);
colorbar;
figure;
imagesc(imageFiltered);
colorbar;
figure;
imagesc(nullMean);
colorbar;
figure;
imagesc(nullStd);
colorbar;