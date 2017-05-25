clc;
close all;
clearvars;

%BLOCK DATA 140316 SCRIPT
%Displays a b/g/w and sample image
%Displays its greyvalue histogram

%load the data
block_data = BlockData_140316();

%get the b/g/w and sample images
black = block_data.loadBlack(1);
grey = block_data.loadGrey(1);
white = block_data.loadWhite(1);
sample = block_data.loadSample(1);

%plot the b/g/w and sample images
fig = figure;
subplot(2,2,1,imagesc_truncate(black));
axis(gca,'off');
colormap gray;
colorbar;
title('Black');
subplot(2,2,2,imagesc_truncate(grey));
axis(gca,'off');
colormap gray;
colorbar;
title('Grey');
subplot(2,2,3,imagesc_truncate(white));
axis(gca,'off');
colormap gray;
colorbar;
title('White');
subplot(2,2,4,imagesc_truncate(sample));
axis(gca,'off');
colormap gray;
colorbar;
title('Sample');
%save it
saveas(fig, 'reports/figures/data/140316_image.png');

%reshape the images into a vector
black = reshape(black,[],1);
grey = reshape(grey,[],1);
white = reshape(white,[],1);
sample = reshape(sample,[],1);

%plot the histogram of the greyvalues
%use each integer as a bin (min : max)
%color in b/g/w/ using r/g/b
fig = figure;
%plot the b/g/w
subplot(2,1,1);
h = histogram(black,min(black):max(black));
h.EdgeColor = 'r';
h.FaceColor = 'r';
xlim([0,6E4]);
hold on;
h = histogram(grey,min(grey):max(grey));
h.EdgeColor = 'g';
h.FaceColor = 'g';
h = histogram(white,min(white):max(white));
h.EdgeColor = 'b';
h.FaceColor = 'b';
legend('Black','Grey','White');
xlabel('Greyvalue');
ylabel('Frequency');
%plot the sample
subplot(2,1,2);
h = histogram(sample,min(sample):max(sample));
h.FaceColor = 'k';
legend('Sample');
xlim([0,6E4]);
xlabel('Greyvalue');
ylabel('Frequency');
%save the figure
saveas(fig, 'reports/figures/data/140316_histo.eps','epsc');