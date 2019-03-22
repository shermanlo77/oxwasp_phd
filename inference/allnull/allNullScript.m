%ALL NULL GAUSSIAN SCRIPT
%Filters a image
%Shows the image before and after filtering
%Shows the empirical null mean and std image
%Plots the p values of the post filter greyvalues

clc;
clearvars;
close all;

imageSize = 256;

%=====ALL NULL GAUSSIAN=====%
randStream = RandStream('mt19937ar','Seed',uint32(3499211588)); %instantise a rng
image = randStream.randn(imageSize,imageSize); %create gaussian image
allNullExample(image, 'allNullGaussianScript');

%=====ALL NULL PLANE=====%
%contaminate the image
randStream = RandStream('mt19937ar','Seed',uint32(581365657)); %instantise a rng
trueNullStd = 2;
trueNullMeanGrad = [0.01, 0.01];
defectSimulator = PlaneMult(randStream, trueNullMeanGrad, trueNullStd);
image = defectSimulator.getDefectedImage([imageSize, imageSize]);
allNullExample(image, 'allNullPlaneScript');

%FUNCTION: ALL NULL EXAMPLE
%PARAMETERS:
  %image: to be filtered
  %name: prefix of the file saved containing the figures
function allNullExample(image, name)
  
  directory = fullfile('reports','figures','inference');
  
  radius = 20;
  filter = EmpiricalNullFilter(radius); %filter it
  filter.filter(image);
  
  %get the empirical null and the filtered image
  imageFiltered = filter.getFilteredImage();
  nullMean = filter.getNullMean();
  nullStd = filter.getNullStd();
  clim = [min(min(imageFiltered)), max(max(imageFiltered))];
  stdClim = [0, clim(2)];
  
  %plot the image before filtering
  fig = LatexFigure.sub();
  imagePlot = ImagescSignificant(image);
  imagePlot.setCLim(clim);
  imagePlot.plot();
  saveas(fig,fullfile(directory, strcat(name,'_beforeFilter.eps')),'epsc');
  
  %plot the image after filtering
  fig = LatexFigure.sub();
  imagePlot = ImagescSignificant(imageFiltered);
  imagePlot.setCLim(clim);
  imagePlot.plot();
  saveas(fig,fullfile(directory, strcat(name,'_afterFilter.eps')),'epsc');
  
  %empirical null mean plot
  fig = LatexFigure.sub();
  imagePlot = ImagescSignificant(nullMean);
  imagePlot.setCLim(clim);
  imagePlot.plot();
  saveas(fig,fullfile(directory, strcat(name,'_nullMean.eps')),'epsc');
  
  %empirical null std plot
  fig = LatexFigure.sub();
  imagePlot = ImagescSignificant(nullStd);
  imagePlot.setCLim(stdClim);
  imagePlot.plot();
  saveas(fig,fullfile(directory, strcat(name,'_nullStd.eps')),'epsc');
  
  %p value plot
  zTester = ZTester(imageFiltered);
  zTester.doTest();
  fig = LatexFigure.sub();
  zTester.plotPValues();
  saveas(fig,fullfile(directory, strcat(name,'_pValue.eps')),'epsc');

end
