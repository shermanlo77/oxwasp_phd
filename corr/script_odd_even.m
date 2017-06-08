clc;
clearvars;
close all

bgw_data = BGW_140316();

black = bgw_data.loadBlack(1);

figure;
imagesc_truncate(black);

y = mean(black,2);
T = numel(y);

figure;
plot(y);

figure;
plot(1:2:T,y(1:2:T));
hold on;
plot(2:2:T,y(2:2:T));
hold off;