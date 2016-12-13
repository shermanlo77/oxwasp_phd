clearvars;

surface_fitting = Polynomial([2048,2048],[1996,1996],16);

%surface_fitting.loadBlack('D:/black');
surface_fitting.loadBlack('/home/sherman/Documents/data/black');
surface_fitting.fitSurface(1,1);
surface_fitting.plotBlackSurface([1,99]);
SurfaceFitting.imagesc_truncate(surface_fitting.getResidualImage(1),[1,99]);
colorbar;
figure;
residual = surface_fitting.getResidualImage(1);
histogram(reshape(residual,[],1));

mse = surface_fitting.crossValidation(1,1:5);
mse = surface_fitting.rotateCrossValidation(1:5);
figure;
boxplot(mse);
xlabel('Order');
ylabel('Mean squared error');

%surface_fitting.loadWhite('D:/white');
surface_fitting.loadWhite('/home/sherman/Documents/data/white');
surface_fitting.fitSurface(1,1);
surface_fitting.plotWhiteSurface([1,99]);
SurfaceFitting.imagesc_truncate(surface_fitting.getResidualImage(1),[1,99]);
colorbar;

mse = surface_fitting.crossValidation(1,1:5);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clearvars;

surface_fitting = Lowess([2048,2048],[1996,1996],16);

surface_fitting.loadBlack('D:/black');
%surface_fitting.loadBlack('/home/sherman/Documents/data/black');
surface_fitting.fitSurface(1,0.25);
surface_fitting.plotBlackSurface([1,99]);
residual = surface_fitting.getResidualImage(2);
figure;
imagesc(residual, prctile(reshape(residual,[],1),[1,99]));
colorbar;
colormap gray;

mse = surface_fitting.crossValidation(1,1:5);
mse = surface_fitting.rotateCrossValidation(1:5);
figure;
boxplot(mse);
xlabel('Order');
ylabel('Mean squared error');

surface_fitting.loadWhite('D:/white');
%surface_fitting.loadWhite('/home/sherman/Documents/data/white');
surface_fitting.fitSurface(1,1);
surface_fitting.plotWhiteSurface([1,99]);
mse = surface_fitting.crossValidation(1,1:5);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clearvars;

surface_fitting = Polynomial([2048,2048],[1996,1996],16);

p_order_black = 1;
p_order_white = 2;
k = 1000;

surface_fitting.loadBlack('D:/black');
%surface_fitting.loadBlack('/home/sherman/Documents/data/black');
surface_fitting.fitSurface(1,p_order_black);
surface_fitting.plotBlackSurface([1,99]);
SurfaceFitting.imagesc_truncate(surface_fitting.getResidualImage(1),[1,99]);
colorbar;

residual_sample = reshape(surface_fitting.getResidualImage(1),[],1);
residual_sample = randsample(residual_sample,k);
figure;
qqplot(residual_sample);
figure;
histogram(residual_sample);

figure;
imagesc(surface_fitting.getZResidual(p_order_black),[-5,5]);
colorbar;
colormap gray;

surface_fitting.loadWhite('D:/white');
%surface_fitting.loadWhite('/home/sherman/Documents/data/white');
surface_fitting.fitSurface(1,p_order_white);
surface_fitting.plotWhiteSurface([1,99]);
SurfaceFitting.imagesc_truncate(surface_fitting.getResidualImage(1),[1,99]);
colorbar;

residual_sample = reshape(surface_fitting.getResidualImage(1),[],1);
residual_sample = randsample(residual_sample,k);
figure;
qqplot(residual_sample);
figure;
histogram(residual_sample);

figure;
imagesc(surface_fitting.getZResidual(p_order_white),[-5,5]);
colorbar;
colormap gray;

surface_fitting.plotBlackShadeCorrect([1,99]);
colorbar;

surface_fitting.plotWhiteShadeCorrect([1,99]);
colorbar;

surface_fitting.clearBWStack();

surface_fitting.loadScan('D:/block');
%surface_fitting.loadScan('/home/sherman/Documents/data/block');

surface_fitting.plotScan(1,[1,99]);
colorbar;

surface_fitting.plotScanShadeCorrect_fit(1,[1,99]);
colorbar;

surface_fitting.plotScanShadeCorrect_train(1,[1,99]);
colorbar;

scan = surface_fitting.scan_stack(:,:,1);
scan_shadingCorrection_fit = surface_fitting.shadeCorrect_fit(scan);
scan_shadingCorrection_train = surface_fitting.shadeCorrect_train(scan);
mean_scan = mean(reshape(scan,[],1));
std_scan = std(reshape(scan,[],1));
scan_shadingCorrection_fit = 2*(scan_shadingCorrection_fit-0.5)*std_scan + mean_scan;
scan_shadingCorrection_train = 2*(scan_shadingCorrection_train-0.5)*std_scan + mean_scan;

SurfaceFitting.imagesc_truncate(scan-scan_shadingCorrection_fit,[1,99]);
colorbar;

SurfaceFitting.imagesc_truncate(scan-scan_shadingCorrection_train,[1,99]);
colorbar;

SurfaceFitting.imagesc_truncate(surface_fitting.white_fit - surface_fitting.black_fit,[1,99]);
colorbar;
SurfaceFitting.imagesc_truncate(surface_fitting.white_train - surface_fitting.black_train,[1,99]);
colorbar;