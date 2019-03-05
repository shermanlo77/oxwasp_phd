clc;
clearvars;
close all;
rng(uint32(1514448035), 'twister');
load('results/subregion1.mat');

p0 = mixture_gaussian.ComponentProportion(1);
mean_null = mixture_gaussian.mu(1);
mean_alt = mixture_gaussian.mu(2);
std_null = sqrt(mixture_gaussian.Sigma(1));
std_alt = sqrt(mixture_gaussian.Sigma(2));

mean_alt_array = linspace(3,5,20);

N1 = round(200*200 * (1-p0));
n_repeat = 20;

grid_array = linspace(100,600,20);
power_array = zeros(1,numel(grid_array));

for i_alt = 1:numel(mean_alt_array)
    
    mean_alt = mean_alt_array(i_alt);
    
    for i_grid = 1:numel(grid_array)

        grid_size = grid_array(i_grid);

        N = round(grid_size^2);
        N0 = N - N1;

        for i_repeat = 1:n_repeat
            z_array = zeros(N,1);

            z_array(1:N0) = normrnd(mean_null,std_null,N0,1);
            z_array((N0+1):end) = normrnd(mean_alt, std_alt,N1,1);

            z_plot = linspace(min(z_array),max(z_array),100);

            z_tester = ZTester(z_array);
            z_tester.estimateNull(100);
            z_tester.doTest();
            z_critical = z_tester.getZCritical();

            p0_hat = z_tester.estimateP0();

            power_array(i_repeat, i_grid, i_alt) = normcdf(z_critical(1),mean_alt,std_alt) + normcdf(z_critical(2),mean_alt,std_alt,'upper');
        end
    end
end

power_array_plot = squeeze(mean(power_array,1));
[alt_plot, grid_plot] = meshgrid(mean_alt_array, grid_array);

figure;
surf(alt_plot,grid_plot, power_array_plot);
xlabel('alternative mean');
ylabel('grid size');
zlabel('power');