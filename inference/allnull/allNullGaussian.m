%SCRIPT: ALL NULL GAUSSIAN
%Filters a Gaussian image
%Shows the image before and after filtering
%Shows the empirical null mean and std image
%Plots the qq plot of the post filter greyvalues

clc;
clearvars;
close all;

directory = fullfile('reports','figures','inference','noContamination');
randStream = RandStream('mt19937ar','Seed',uint32(3499211588)); %instantise a rng

imageSize = 256; %size of the image
radius = 20; %radius of kernel

%save the imageSize and radius
file_id = fopen(fullfile(directory,'imageSize.txt'),'w');
fprintf(file_id,'%d',imageSize);
fclose(file_id);
file_id = fopen(fullfile(directory,'radius.txt'),'w');
fprintf(file_id,'%d',radius);
fclose(file_id);

image = randStream.randn(imageSize,imageSize); %create gaussian image
filter = EmpiricalNullFilter(radius); %filter it
filter.setNInitial(3);
filter.filter(image);

%get the empirical null and the filtered image
imageFiltered = filter.getFilteredImage();
nullMean = filter.getNullMean();
nullStd = filter.getNullStd();

%plot the image before filtering
fig = LatexFigure.sub();
imagePlot = ImagescSignificant(image);
imagePlot.setCLim([-3,3]);
imagePlot.plot();
saveas(fig,fullfile(directory, 'allNullGaussianunfiltered.eps'),'epsc');

%plot the image after filtering
fig = LatexFigure.sub();
imagePlot = ImagescSignificant(imageFiltered);
imagePlot.setCLim([-3,3]);
imagePlot.plot();
saveas(fig,fullfile(directory, 'allNullGaussianfiltered.eps'),'epsc');

%qq plot of the image after filtering
fig = LatexFigure.sub();
qqplot(reshape(imageFiltered,[],1));
title('');
ylabel('quantiles of filtered pixels');
xlabel('standard normal quantiles');
saveas(fig,fullfile(directory, 'allNullGaussianqq.eps'),'epsc');

%empirical null plot
fig = LatexFigure.sub();
imagePlot = ImagescSignificant(nullMean);
imagePlot.plot();
saveas(fig,fullfile(directory, 'allNullGaussiannullmean.eps'),'epsc');

fig = LatexFigure.sub();
imagePlot = ImagescSignificant(nullStd);
imagePlot.plot();
saveas(fig,fullfile(directory, 'allNullGaussiannullstd.eps'),'epsc');