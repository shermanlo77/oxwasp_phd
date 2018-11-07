%SCRIPT: ALL NULL PLANE
%Filters a Gaussian image x scale + gradient
%Shows the empirical null mean and std image
%Plots the qq plot of the post filter greyvalues

clc;
clearvars;
close all;

randStream = RandStream('mt19937ar','Seed',uint32(676943031)); %instantise a rng

imageSize = 256;
radius = 20; %radius of kernel
trueNullStd = 2;
trueNullMeanGrad = [0.01, 0.01];
altP = 0.1;
altMean = 2;
altStd = 1;
defectSimulator = PlaneMultDust(randStream, trueNullMeanGrad, trueNullStd, altP, altMean, altStd);
[image, isAltImage, imagePreBias] = defectSimulator.getDefectedImage([imageSize, imageSize]);

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

%get the min and max z value
zmin = min([min(min(imagePreBias)), min(min(imageFiltered))]);
zmax = max([max(max(imagePreBias)), max(max(imageFiltered))]);

%plot the image pre/post bias with significant pixels highlighted
zTesterPreBias = ZTester(imagePreBias);
zTesterPreBias.doTest();
figure;
imagePlot = ImagescSignificant(imagePreBias);
imagePlot.addSigPixels(zTesterPreBias.sig_image);
imagePlot.setCLim([zmin,zmax]);
imagePlot.plot();

zTesterPostBias = ZTester(imageFiltered);
zTesterPostBias.doTest();
figure;
imagePlot = ImagescSignificant(imageFiltered);
imagePlot.addSigPixels(zTesterPostBias.sig_image);
imagePlot.setCLim([zmin,zmax]);
imagePlot.plot();

%empirical null plot
figure;
imagesc(nullMean);
colorbar;
figure;
imagesc(nullStd);
colorbar;

%work out true and false positive before and after bias adding
%print out the area under the roc
[falsePositivePreBias, truePositivePreBias, areaRocPreBias] = roc(imagePreBias, isAltImage, 100);
[falsePositivePostBias, truePositivePostBias, areaRocPostBias] = ...
    roc(imageFiltered, isAltImage, 100);
figure;
plot(falsePositivePreBias, truePositivePreBias);
hold on;
plot(falsePositivePostBias, truePositivePostBias);
plot([0,1],[0,1],'k--');
xlabel('false positive rate');
ylabel('true positive rate');
legend('pre bias adding','post bias adding','Location','southeast');
disp(strcat('pre bias roc area = ',num2str(areaRocPreBias)));
disp(strcat('post bias roc area = ',num2str(areaRocPostBias)));

%for this particular significant level, print the type 1 and type 2 error
%type 1 = false positive
%type 2 = false negative
disp('For pre bias adding');
type1Error = sum(sum(zTesterPreBias.sig_image(~isAltImage))) / sum(sum(~isAltImage));
type2Error = sum(sum(~(zTesterPreBias.sig_image(isAltImage)))) / sum(sum(isAltImage));
disp('type 1 error');
disp(type1Error);
disp('type 2 error');
disp(type2Error);

disp('For post bias adding');
type1Error = sum(sum(zTesterPostBias.sig_image(~isAltImage))) / sum(sum(~isAltImage));
type2Error = sum(sum(~(zTesterPostBias.sig_image(isAltImage)))) / sum(sum(isAltImage));
disp('type 1 error');
disp(type1Error);
disp('type 2 error');
disp(type2Error);