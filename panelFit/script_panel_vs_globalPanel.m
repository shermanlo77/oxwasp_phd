%SCRIPT PANEL-WISE VS GLOBAL + PANEL-WISE 2nd ORDER POLYNOMIAL FIT
%Fits a panel-wise 2nd order polynomial and a global+panel-wise 2nd order
%polynomial on a white image. Plots the residuals for both of these fit,
%they are almost the same, suggesting the addition of the global polynomial
%didn't add much

clc;
clearvars;
close all;

%load the data
bgw_data = BGW_140316();

%get the white image
white = bgw_data.loadWhite(1);

%for the panel-wise fit, then the global+panel-wise fit
for i = 1:2
    
    %instantise the corresponding polynomial fitter
    if i == 1
        polynomial_fitter = PanelPolynomialFitter(bgw_data.panel_counter);
    else
        polynomial_fitter = GlobalPanelPolynomialFitter(bgw_data.panel_counter);
    end
    
    %fit the polynomial on the white image
    polynomial_fitter.fitPolynomial(white);
    %get the fitted polynomial
    white_predict = polynomial_fitter.fitted_image;

    %plot the residual of the fitted polynomial
    figure;
    imagesc_truncate(white - white_predict);
    colorbar;
    axis(gca,'off');
end