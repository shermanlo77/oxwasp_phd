clearvars;

surface_fitting = Polynomial([2048,2048],[1996,1996],16);

%surface_fitting.loadBlack('D:/black');
surface_fitting.loadBlack('/home/sherman/Documents/data/black');
surface_fitting.fitSurface(1,1);
surface_fitting.plotBlackSurface([1,99]);
SurfaceFitting.imagesc_truncate(surface_fitting.getResidualImage(1),[1,99]);
colorbar;

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

p_order = 1;

%surface_fitting.loadBlack('D:/black');
surface_fitting.loadBlack('/home/sherman/Documents/data/black');
surface_fitting.fitSurface(1,p_order);
surface_fitting.plotBlackSurface([1,99]);

%surface_fitting.loadWhite('D:/white');
surface_fitting.loadWhite('/home/sherman/Documents/data/white');
surface_fitting.fitSurface(1,p_order);
surface_fitting.plotWhiteSurface([1,99]);

surface_fitting.plotBlackShadeCorrect([1,99]);
colorbar;

surface_fitting.plotWhiteShadeCorrect([1,99]);
colorbar;

surface_fitting.clearBWStack();

surface_fitting.loadScan('/home/sherman/Documents/data/block');

surface_fitting.plotScan(1,[0,100]);

surface_fitting.plotScanShadeCorrect_fit(1,[0,100]);

surface_fitting.plotScanShadeCorrect_train(1,[1,99]);
