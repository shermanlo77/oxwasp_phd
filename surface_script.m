clearvars;

surface_fitting = Polynomial([2048,2048],[1996,1996],16);

%surface_fitting.loadBlack('D:/black');
surface_fitting.loadBlack('/home/sherman/Documents/data/black');
surface_fitting.fitSurface(1,1);
surface_fitting.plotBlackSurface([1,99]);
residual = surface_fitting.getResidualImage(1);
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

%surface_fitting.loadWhite('D:/white');
surface_fitting.loadWhite('/home/sherman/Documents/data/white');
surface_fitting.fitSurface(1,1);
surface_fitting.plotWhiteSurface([1,99]);
residual = surface_fitting.getResidualImage(1);
figure;
imagesc(residual, prctile(reshape(residual,[],1),[1,99]));
colorbar;
colormap gray;

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