clc;
close all;
clearvars;

%Displays images from the abs block scan
abs_block = AbsBlock_Mar16();
figure;
imagesc(abs_block.loadImage(100));
colormap gray;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

bgw = Bgw_Mar16();
for i = 1:numel(bgw.reference_scan_array)
    figure;
    imagesc_truncate(bgw.reference_scan_array(i).loadImage(20));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

abs_block = AbsBlock_July16_30deg();
figure;
subplot(2,1,1);
imagesc(abs_block.loadImage(100));
colormap gray;
subplot(2,1,2);
imagesc(abs_block.getARTistImage());
colormap gray;

abs_block = AbsBlock_July16_120deg();
figure;
subplot(2,1,1);
imagesc(abs_block.loadImage(100));
colormap gray;
subplot(2,1,2);
imagesc(abs_block.getARTistImage());
colormap gray;

for i = 2:numel(abs_block.reference_scan_array)
    figure;
    subplot(2,1,1);
    imagesc(abs_block.reference_scan_array(i).loadImage(20));
    subplot(2,1,2);
    imagesc(abs_block.reference_scan_array(i).getARTistImage());
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

abs_block = AbsBlock_Sep16_30deg();
figure;
subplot(2,1,1);
imagesc(abs_block.loadImage(20));
colormap gray;
subplot(2,1,2);
imagesc(abs_block.getARTistImage());
colormap gray;

abs_block = AbsBlock_Sep16_120deg();
figure;
subplot(2,1,1);
imagesc(abs_block.loadImage(20));
colormap gray;
subplot(2,1,2);
imagesc(abs_block.getARTistImage());
colormap gray;

for i = 2:numel(abs_block.reference_scan_array)
    figure;
    subplot(2,1,1);
    imagesc(abs_block.reference_scan_array(i).loadImage(20));
    subplot(2,1,2);
    imagesc(abs_block.reference_scan_array(i).getARTistImage());
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