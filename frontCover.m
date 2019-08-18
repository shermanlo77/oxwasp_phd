%MIT License
%Copyright (c) 2019 Sherman Lo

%REQUIRES DefectDetectSubRoiAbsFilterDeg120

close all;
clearvars;

masterImage = zeros(2000, 2000, 3);

scan = AbsFilterDeg120();
projection = scan.loadImage(1);

fig = LatexFigure.main();
imagesc = Imagesc(projection);
projectionImage = imagesc.plot();
clim = imagesc.clim;
masterImage(:, 1:1000, :) = projectionImage.CData(:, 1:1000, :);

experiment = DefectDetectSubRoiAbsFilterDeg120();
zTester = ZTester(experiment.zFilterArray(:,:,end));
zTester.doTest();
logp = -log10(zTester.pImage);

fig = LatexFigure.main();
imagesc = Imagesc(logp);
imagesc.setCLim([0,15]);
projectionImage = imagesc.plot();
masterImage(:, 1001:end, :) = projectionImage.CData(:, 1001:end, :);

segmentation = repmat(scan.getSegmentation(),1,1,3);
masterImage(~segmentation) = 1;

imwrite(masterImage,fullfile('reports','figures','frontCover.jpg'));