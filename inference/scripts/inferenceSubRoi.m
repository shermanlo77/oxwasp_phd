%SUB ROI SCRIPT
%REQUIRES: DefectDetectAbsFilterDeg120
%Provide figures to justify the use of sub roi
%
%Get z image for the AbsFilter at 120
%Highlight bottom right corner and plot histogram of test statistics at that corner
%
%Plots the sub segmentation of the dataset

clearvars;
close all;

%get z image
experiment = DefectDetectAbsFilterDeg120();
scan = experiment.scan();
zImage = experiment.zImage;

%where of the z image to take sample
xArray = [1500,1745];
yArray = [1660,1900];

%plot the z image with the sub-sample highlighed
fig = LatexFigure.sub();
axis xy;
imagesc = Imagesc(zImage);
imagesc.plot();
hold on;
plot(xArray,yArray(1)*ones(1,2),'r--');
plot(xArray,yArray(2)*ones(1,2),'r--');
plot(xArray(1)*ones(1,2),yArray,'r--');
plot(xArray(2)*ones(1,2),yArray,'r--');
saveas(fig, fullfile('reports','figures','inference','cornerSelect.eps'),'epsc');

%plot histogram of the z statistics
zSub = zImage(yArray(1):yArray(2), xArray(1):xArray(2));
zTester = ZTester(zSub);
fig = LatexFigure.sub();
zTester.plotHistogram();
saveas(fig, fullfile('reports','figures','inference','cornerSelectHist.eps'),'epsc');

%plot the sub segments
fig = LatexFigure.sub();
axis xy;
imagesc = Imagesc(zImage);
imagesc.plot();
hold on;
for iSegmentation = 1:scan.nSubSegmentation
  segmentation = scan.getSubSegmentation(iSegmentation);
  boundary = bwboundaries(segmentation);
  boundary = boundary{1};
  plot(boundary(:,2), boundary(:,1), 'r:', 'LineWidth', 2.5);
end
saveas(fig, fullfile('reports','figures','inference','segment.eps'),'epsc');
saveas(fig, fullfile('reports','figures','inference','segment.tiff'),'tiff');
