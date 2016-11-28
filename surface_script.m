clearvars;

surface_fitting = SurfaceFitting([2048,2048],[1996,1996],16);
surface_fitting.loadBlack('/home/sherman/Documents/data/black');

surface_fitting.fitPolynomialPanel_black(1,1);
%surface_fitting.clearBlack();
surface_fitting.plotPolynomialPanel_black([1,99]);