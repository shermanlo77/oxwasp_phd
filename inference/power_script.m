clc;
clearvars;
close all;
rng(uint32(2705234785), 'twister');
load('results/subregion1.mat');

p0 = mixture_gaussian.ComponentProportion(1);
mean_null = mixture_gaussian.mu(1);
mean_alt = mixture_gaussian.mu(2);
std_null = sqrt(mixture_gaussian.Sigma(1));
std_alt = sqrt(mixture_gaussian.Sigma(2));


N1 = round(200*200 * (1-p0));
n_repeat = 100;

p0_array = linspace(0.7,0.99,10);
power_array = zeros(1,numel(p0_array));

for i_p0 = 1:numel(p0_array)
    
    p0 = p0_array(i_p0);
    
    N0 = round(p0*N1/(1-p0));
    N = N0 + N1;
    
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

        if i_repeat == 1
            figure;
            histogram(z_array,'Normalization','CountDensity','DisplayStyle','stairs');
            hold on;
            plot(z_plot,N0*normpdf(z_plot,mean_null,std_null));
            plot(z_plot,N1*normpdf(z_plot,mean_alt,std_alt));
        %     plot(z_plot,N*z_tester.density_estimator.getDensityEstimate(z_plot));
        %     plot(z_plot,N*p0_hat*normpdf(z_plot,z_tester.mean_null,z_tester.std_null));
        %     plot(z_plot,N*(1-p0_hat)*z_tester.estimateH1Density(z_plot));
            plot([z_critical(1),z_critical(1)],[0,N0*normpdf(0)],'r-','LineWidth',2);
            plot([z_critical(2),z_critical(2)],[0,N0*normpdf(0)],'r-','LineWidth',2);
            xlabel('z statistic');
            ylabel('frequency density');
            legend('histogram','null','alt');
        end

        power_array(i_repeat, i_p0) = normcdf(z_critical(1),mean_alt,std_alt) + normcdf(z_critical(2),mean_alt,std_alt,'upper');
    end
end

figure;
boxplots = Boxplots(power_array, true);
boxplots.setPosition(p0_array);
boxplots.setWhiskerCap(false);
boxplots.plot();
xlabel('p_0');
ylabel('power');