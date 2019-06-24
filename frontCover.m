clc;
close all;
clearvars;

masterImage = zeros(2000, 2000, 3);
dilateSize = 3;

scan = AbsFilterDeg120();
projection = scan.loadImage(1);
projectionImagesc = Imagesc(projection);
projectionImage = projectionImagesc.plot();
clim = projectionImagesc.clim;
masterImage(1001:end, 1:1000, :) = projectionImage.CData(1001:end, 1:1000, :);

projection = scan.calibrationScanArray(scan.whiteIndex).loadImage(1);
projectionImagesc = Imagesc(projection);
projectionImagesc.setCLim(clim);
projectionImage = projectionImagesc.plot();
masterImage(1001:end, 1001:end, :) = projectionImage.CData(1001:end, 1001:end, :);


[test, artist, zImage] = inferenceExample();
zTester = ZTester(zImage);
zTester.doTest();

scan.addShadingCorrectorLinear();
projection = scan.loadImage(1);
projectionImagesc = Imagesc(projection);
projectionImagesc.setDilateSize(dilateSize);
projectionImagesc.setCLim(clim);
projectionImagesc.addPositivePixels(zTester.positiveImage);
projectionImage = projectionImagesc.plot();
masterImage(1:1000, 1001:end, :) = projectionImage.CData(1:1000, 1001:end, :);


experiment = DefectDetectSubRoiAbsFilterDeg120();
zTester = ZTester(experiment.zFilterArray(:,:,end));
zTester.doTest();

projectionImagesc = Imagesc(projection);
projectionImagesc.setDilateSize(dilateSize);
projectionImagesc.setCLim(clim);
projectionImagesc.addPositivePixels(zTester.positiveImage);
projectionImage = projectionImagesc.plot();
masterImage(1:1000, 1:1000, :) = projectionImage.CData(1:1000, 1:1000, :);

imwrite(masterImage,'frontCover.png');