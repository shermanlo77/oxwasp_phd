clearvars;

surface_fitting = SurfaceFitting([2048,2048],[1996,1996],16);

surface_fitting.loadBlack('/home/sherman/Documents/data/black');
surface_fitting.fitPolynomialPanel(1,1);
surface_fitting.plotPolynomialPanel_black([1,99]);
disp(surface_fitting.crossValidation(1));

surface_fitting.loadWhite('/home/sherman/Documents/data/white');
surface_fitting.fitPolynomialPanel(1,1);
surface_fitting.plotPolynomialPanel_white([1,99]);
disp(surface_fitting.crossValidation(1));