clc;
close all;
clearvars;

data = AbsBlock_Sep16_120deg();
clim = [2.5E4,6E4];

%Displays images from the abs block scan
fig = LatexFigure.sub();
imagescPlot = Imagesc(data.loadImage(1));
imagescPlot.setCLim(clim);
imagescPlot.plot();

fig = LatexFigure.sub();
imagescPlot = Imagesc(data.referenceScanArray(1).loadImage(1));
%imagescPlot.setCLim(clim);
imagescPlot.plot();

fig = LatexFigure.sub();
imagescPlot = Imagesc(data.referenceScanArray(3).loadImage(1));
imagescPlot.setCLim(clim);
imagescPlot.plot();

fig = LatexFigure.sub();
imagescPlot = Imagesc(data.referenceScanArray(5).loadImage(1));
imagescPlot.setCLim(clim);
imagescPlot.plot();

nSample = 100;
xArray = zeros(nSample*data.getNReference(),1);
yArray = zeros(data.getNReference(),1);
referenceArray = zeros(data.height, data.width, data.getNReference());
for i = 1:data.getNReference
  referenceArray(:,:,i) = data.referenceScanArray(i).loadImage(1);
  yArray(i) = mean(reshape(referenceArray(:,:,i),[],1)); 
end
yArray = repmat(yArray,nSample,1);
for i = 1:nSample
  position = [randi([1,data.height]),randi([1,data.width])];
  for j = 1:data.getNReference()
    xArray((i-1)*data.getNReference + j) =  referenceArray(position(1),position(2),j);
  end
end

fig = LatexFigure.sub();
xlim([0,6E4]);
ylim([0,6E4]);
ylabel('within image mean');
xlabel('grey value');
hold on;

index = 1:data.getNReference();
ax = scatter(xArray(index),yArray(index));
p = polyfit(xArray(index), yArray(index) ,1);
line([0,6E4],[p(2), p(2)+6E4*p(1)], 'Color', ax.CData);

index = data.getNReference()+1 : 2*data.getNReference();
ax = scatter(xArray(index),yArray(index));
p = polyfit(xArray(index), yArray(index) ,1);
line([0,6E4],[p(2), p(2)+6E4*p(1)], 'Color', ax.CData);

index = 2*data.getNReference()+1 : 3*data.getNReference();
ax = scatter(xArray(index),yArray(index));
p = polyfit(xArray(index), yArray(index) ,1);
line([0,6E4],[p(2), p(2)+6E4*p(1)], 'Color', ax.CData);

fig = LatexFigure.sub();
xlim([0,6E4]);
ylim([0,6E4]);
ylabel('within image mean');
xlabel('grey value');
hold on;
for i = 1:nSample
  index = (i-1)*data.getNReference()+1 : i*data.getNReference();
  ax = scatter(xArray(index),yArray(index));
  p = polyfit(xArray(index), yArray(index) ,1);
  line([0,6E4],[p(2), p(2)+6E4*p(1)], 'Color', ax.CData);
end

data.addLinearShadingCorrector();
fig = LatexFigure.sub();
imagescPlot = Imagesc(data.loadImage(1));
imagescPlot.setCLim(clim);
imagescPlot.plot();


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

bgw = Bgw_Mar16();
for i = 1:numel(bgw.reference_scan_array)
    figure;
    imagesc_truncate(bgw.reference_scan_array(i).loadImage(20));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data = AbsBlock_July16_30deg();
figure;
subplot(2,1,1);
imagesc(data.loadImage(100));
colormap gray;
subplot(2,1,2);
imagesc(data.getARTistImage());
colormap gray;

data = AbsBlock_July16_120deg();
figure;
subplot(2,1,1);
imagesc(data.loadImage(100));
colormap gray;
subplot(2,1,2);
imagesc(data.getARTistImage());
colormap gray;

for i = 2:numel(data.reference_scan_array)
    figure;
    subplot(2,1,1);
    imagesc(data.reference_scan_array(i).loadImage(20));
    subplot(2,1,2);
    imagesc(data.reference_scan_array(i).getARTistImage());
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data = AbsBlock_Sep16_30deg();
figure;
subplot(2,1,1);
imagesc(data.loadImage(20));
colormap gray;
subplot(2,1,2);
imagesc(data.getARTistImage());
colormap gray;

data = AbsBlock_Sep16_120deg();
figure;
subplot(2,1,1);
imagesc(data.loadImage(20));
colormap gray;
subplot(2,1,2);
imagesc(data.getARTistImage());
colormap gray;

for i = 2:numel(data.reference_scan_array)
    figure;
    subplot(2,1,1);
    imagesc(data.reference_scan_array(i).loadImage(20));
    subplot(2,1,2);
    imagesc(data.reference_scan_array(i).getARTistImage());
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

titanium_block = TitaniumBlock_Dec16_30deg();
figure;
subplot(2,1,1);
imagesc(titanium_block.loadImage(20));
colormap gray;
subplot(2,1,2);
imagesc(titanium_block.getARTistImage());
colormap gray;

titanium_block = TitaniumBlock_Dec16_120deg();
figure;
subplot(2,1,1);
imagesc(titanium_block.loadImage(20));
colormap gray;
subplot(2,1,2);
imagesc(titanium_block.getARTistImage());
colormap gray;

for i = 2:numel(titanium_block.reference_scan_array)
    figure;
    subplot(2,1,1);
    imagesc(titanium_block.reference_scan_array(i).loadImage(20));
    subplot(2,1,2);
    imagesc(titanium_block.reference_scan_array(i).getARTistImage());
end