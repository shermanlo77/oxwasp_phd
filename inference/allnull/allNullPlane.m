%SCRIPT: ALL NULL PLANE
%Filters a Gaussian image x scale + gradient
%Shows the empirical null mean and std image
%Plots the qq plot of the post filter greyvalues

clc;
clearvars;
close all;

directory = fullfile('reports','figures','inference','contamination');
randStream = RandStream('mt19937ar','Seed',uint32(581365657)); %instantise a rng

imageSize = 256;
radius = 20; %radius of kernel
trueNullStd = 2;
trueNullMeanGrad = [0.01, 0.01];

%save the imageSize and radius
file_id = fopen(fullfile(directory,'imageSize.txt'),'w');
fprintf(file_id,'%d',imageSize);
fclose(file_id);
file_id = fopen(fullfile(directory,'radius.txt'),'w');
fprintf(file_id,'%d',radius);
fclose(file_id);

%save the plane gradient and the null std
file_id = fopen(fullfile(directory,'grady.txt'),'w');
fprintf(file_id,'%.2f',trueNullMeanGrad(1));
fclose(file_id);
file_id = fopen(fullfile(directory,'gradx.txt'),'w');
fprintf(file_id,'%.2f',trueNullMeanGrad(2));
fclose(file_id);
file_id = fopen(fullfile(directory,'nullstd.txt'),'w');
fprintf(file_id,'%d',trueNullStd);
fclose(file_id);

%contaminate the image
defectSimulator = PlaneMult(randStream, trueNullMeanGrad, trueNullStd);
image = defectSimulator.getDefectedImage([imageSize, imageSize]);
filter = MadModeNullFilter(radius); %filter it
filter.setNInitial(3);
filter.filter(image);

%get the empirical null and the filtered image
imageFiltered = filter.getFilteredImage();
nullMean = filter.getNullMean();
nullStd = filter.getNullStd();

%qq plot of the image after filtering
fig = LatexFigure.sub();
qqplot(reshape(imageFiltered,[],1));
title('');
ylabel('quantiles of filtered pixels');
xlabel('standard normal quantiles');
saveas(fig,fullfile(directory, 'allNullPlaneqq.eps'),'epsc');

%plot the image pre/post filter with significant pixels highlighted
zTester = ZTester(image);
zTester.doTest();
fig = LatexFigure.sub();
imagePlot = ImagescSignificant(image);
imagePlot.addSigPixels(zTester.sig_image);
imagePlot.plot();
saveas(fig,fullfile(directory, 'allNullPlaneunfiltered.eps'),'epsc');

zTester = ZTester(imageFiltered);
zTester.doTest();
fig = LatexFigure.sub();
imagePlot = ImagescSignificant(imageFiltered);
imagePlot.addSigPixels(zTester.sig_image);
imagePlot.plot();
saveas(fig,fullfile(directory, 'allNullPlanefiltered.eps'),'epsc');

%empirical null plot
fig = LatexFigure.sub();
imagePlot = ImagescSignificant(nullMean);
imagePlot.plot();
saveas(fig,fullfile(directory, 'allNullPlanenullmean.eps'),'epsc');

fig = LatexFigure.sub();
imagePlot = ImagescSignificant(nullStd);
imagePlot.setCLim([0,5]);
imagePlot.plot();
saveas(fig,fullfile(directory, 'allNullPlanenullstd.eps'),'epsc');