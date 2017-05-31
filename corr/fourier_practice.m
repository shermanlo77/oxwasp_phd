clc;
clearvars;
close all;

L = 1000;
t = 0:(L-1);
f_1 = 1/10;
f_2 = 1/30;

x = 10*sin(2*pi*f_1*t) + 5*cos(2*pi*f_2*t+1) + 2 + normrnd(0,5,1,L);

freq = 0:(1/L):1/2;

figure;
plot(x);

x_fft = fft(x);
x_fft = x_fft(1:L/2+1);
x_fft = abs(x_fft/L);
x_fft(2:end-1) = 2*x_fft(2:end-1);
figure;
plot(freq,(x_fft));


% x_fft = fft_leastSquares(x);
% figure;
% plot(freq,(x_fft));
% 
% 
% 
% figure;
% periodogram(x,rectwin(L),L,1);
% 
% c = xcorr(x);
% 
% figure;
% plot(c);
% 
% c_fft = fft(c);
% 
% figure;
% plot(abs(c_fft)/L);
