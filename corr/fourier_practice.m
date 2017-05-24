L = 1000;
t = 0:(L-1);
f_1 = 1/10;
f_2 = 1/30;

x = 10*sin(2*pi*f_1*t) + 8*cos(2*pi*f_2*t+1);

figure;
plot(x);

%x_fft = fft(x);
x_fft = fft_leastSquares(x);

figure;
%plot(2*abs(x_fft/L)); %to get coefficient
plot((2*abs(x_fft/L)).^2);

c = xcorr(x);

figure;
plot(c);

c_fft = fft(c);

figure;
plot(abs(c_fft)/L);
