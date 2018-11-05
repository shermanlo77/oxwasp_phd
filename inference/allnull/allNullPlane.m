%SCRIPT: ALL NULL PLANE
%Filters a Gaussian image x scale + gradient
%Shows the empirical null mean and std image
%Plots the qq plot of the post filter greyvalues

clc;
clearvars;
close all;

randStream = RandStream('mt19937ar','Seed',uint32(581365657)); %instantise a rng

imageSize = 256;
radius = 20; %radius of kernel
trueNullStd = 2;
trueNullMeanGrad = 0.01;
defectSimulator = DefectSimulator([imageSize, imageSize]);

image = randStream.randn(256,256); %create gaussian image
image = image * trueNullStd;
defectSimulator.addPlane([trueNullMeanGrad,trueNullMeanGrad]);
image = defectSimulator.defectImage(image);
filter = EmpiricalNullFilter(radius); %filter it
filter.setNInitial(3);
filter.filter(image);

%get the empirical null and the filtered image
imageFiltered = filter.getFilteredImage();
nullMean = filter.getNullMean();
nullStd = filter.getNullStd();

%qq plot
figure;
qqplot(reshape(imageFiltered,[],1));

%plot the image pre/post filter with significant pixels highlighted
zTester = ZTester(image);
zTester.doTest();
figure;
imagePlot = ImagescSignificant(image);
imagePlot.addSigPixels(zTester.sig_image);
imagePlot.plot();

zTester = ZTester(imageFiltered);
zTester.doTest();
figure;
imagePlot = ImagescSignificant(imageFiltered);
imagePlot.addSigPixels(zTester.sig_image);
imagePlot.plot();

%empirical null plot
figure;
imagesc(nullMean);
colorbar;
figure;
imagesc(nullStd);
colorbar;