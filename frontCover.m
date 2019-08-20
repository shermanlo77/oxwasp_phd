%MIT License
%Copyright (c) 2019 Sherman Lo

%Export the image for the front cover of the thesis
%REQUIRES DefectDetectSubRoiAbsFilterDeg120
%
%Left: shows the raw x-ray projection
%Right: shows the -log p values

close all;
clearvars;

%master image, image to show
  %dim 1 and dim 2: image
  %dim 3: red, green, blue
masterImage = zeros(2000, 2000, 3);

%left hand side of the image
scan = AbsFilterDeg120();
projection = scan.loadImage(1);
fig = LatexFigure.main(); %required to hide the plot
imagesc = Imagesc(projection);
projectionImage = imagesc.plot();
masterImage(:, 1:1000, :) = projectionImage.CData(:, 1:1000, :);

%right hand side of the image
%use the result from an experiment to get the p values
experiment = DefectDetectSubRoiAbsFilterDeg120();
zTester = ZTester(experiment.zFilterArray(:,:,end));
zTester.doTest();
logp = -log10(zTester.pImage);
fig = LatexFigure.main(); %required to hide the plot
imagesc = Imagesc(logp);
imagesc.setCLim([0,15]);
pValueImage = imagesc.plot();
masterImage(:, 1001:end, :) = pValueImage.CData(:, 1001:end, :);

%crop the image using the segmentation
segmentation = repmat(scan.getSegmentation(),1,1,3);
masterImage(~segmentation) = 1;

%save the image
imwrite(masterImage,fullfile('reports','figures','frontCover.jpg'));