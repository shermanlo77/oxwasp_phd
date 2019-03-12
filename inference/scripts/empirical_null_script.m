%EMPIRICAL NULL SCRIPT
%Takes a sample from a z image, containing all null
%Plots a histogram of the z statistics before any empirical null correction
%Does the empirical null correction and print results

clc;
close all;
clearvars;

inferenceExample;

%define the coordinates of the subsample
row_subsample = 1100:1299; %array of row indicies
col_subsample = 400:599; %array of column indicies

%FIGURE
%Plot the z image with a rectangle highlighting the subsample
fig = LatexFigure.sub();
image_plot = ImagescSignificant(z_image);
image_plot.plot();
hold on;
rectangle('Position',[col_subsample(1), row_subsample(1), col_subsample(end)-col_subsample(1)+1, row_subsample(end)-row_subsample(1)+1],'EdgeColor','r','LineStyle','--');
saveas(fig,fullfile('reports','figures','inference','empirical_null_sub_z_image.eps'),'epsc');

%get the subsample of z statistics and do BH multiple hypothesis testing
z_sample = reshape(z_image(row_subsample, col_subsample),[],1);
z_tester = ZTester(z_sample);
z_tester.doTest();
z_bh = z_tester.getZCritical();

%SAVE VALUE
%Save the critical boundary for the BH procedure
file_id = fopen(fullfile('reports','tables','empirical_null_sub_boundary.txt'),'w');
fprintf(file_id,'%.2f',z_bh(2));
fclose(file_id);

%FIGURE
%Plot the histogram of the z statistics
%Plot the histogram of the z statistics with the BH critical boundary
fig = LatexFigure.sub();
z_tester.plotHistogram2(false);
saveas(fig,fullfile('reports','figures','inference','empirical_null_sub_z_histo.eps'),'epsc');

%estimate the empirical null
z_tester.estimateNull(0, int32(1351727940));
mu_0 = z_tester.nullMean; %get the empirical null mean
sigma_0 = z_tester.nullStd; %get the empirical null std
%define what values of x to plot the density estimate
x_plot = linspace(min(z_sample),max(z_sample),500);
parzen = Parzen(reshape(z_sample,[],1));
f_hat = parzen.getDensityEstimate(x_plot); %get the freqency density estimate

%FIGURE
%Plot the frequency density estimate along with the empirical null mean and std
fig = LatexFigure.sub(); 
plot(x_plot,numel(z_sample)*f_hat); %plot density estimate
hold on;
densityAtMode = parzen.getDensityEstimate(mu_0);
plot([mu_0,mu_0],[0,numel(z_sample)*densityAtMode],'k--'); %plot mode
x_plotNull = x_plot((x_plot <  mu_0+sigma_0) & (x_plot >  mu_0-sigma_0));
nullPdfPlot = numel(z_sample) * densityAtMode* normpdf(x_plotNull, mu_0, sigma_0) * sqrt(2*pi) * sigma_0;
plot(x_plotNull,  nullPdfPlot, 'r-.');
%get the value of the density estimate at mu +/- sigma
f_at_sigma_1 = mean(nullPdfPlot([1,end]));
%plot line and arrows to represent 2 std
offset = 0.1;
plot([mu_0-sigma_0,mu_0+sigma_0],f_at_sigma_1*ones(1,2),'k--');
scatter(mu_0-sigma_0+offset,f_at_sigma_1,'k<','filled');
scatter(mu_0+sigma_0-offset,f_at_sigma_1,'k>','filled');
ylabel('frequency density');
xlabel('z stat');
saveas(fig,fullfile('reports','figures','inference','empirical_null_sub_z_parzen.eps'),'epsc');

%SAVE VALUE
%save the empirical null mean
file_id = fopen(fullfile('reports','tables','empirical_null_sub_null_mu.txt'),'w');
fprintf(file_id,'%.2f',mu_0);
fclose(file_id);

%SAVE VALUE
%save the empirical null std
file_id = fopen(fullfile('reports','tables','empirical_null_sub_null_sigma.txt'),'w');
fprintf(file_id,'%.2f',sigma_0);
fclose(file_id);

%Do hypothesis test, corrected using the empirical null
z_tester.doTest();
%get the critical boundary
z_bh_empirical = z_tester.getZCritical();

%SAVE VALUE
%Save the lower critical boundary for the empirical null BH procedure
file_id = fopen(fullfile('reports','tables','empirical_null_sub_null_critical1.txt'),'w');
fprintf(file_id,'%.2f',z_bh_empirical(1));
fclose(file_id);

%SAVE VALUE
%Save the upper critical boundary for the empirical null BH procedure
file_id = fopen(fullfile('reports','tables','empirical_null_sub_null_critical2.txt'),'w');
fprintf(file_id,'%.2f',z_bh_empirical(2));
fclose(file_id);

%SAVE VALUE
%Save the standarised critical boundary for the empirical null BH procedure
z_critical = norminv(1-z_tester.sizeCorrected/2);
file_id = fopen(fullfile('reports','tables','empirical_null_sub_null_critical_zeta.txt'),'w');
fprintf(file_id,'%.2f',z_critical);
fclose(file_id);

%FIGURE
%Plot the histogram of z statistics
%Also plot the empirical null BH critical boundary
fig = LatexFigure.sub();
ax = gca;
z_tester.plotHistogram2(false);
saveas(fig,fullfile('reports','figures','inference','empirical_null_sub_z_histo_null.eps'),'epsc');

%FIGURE
%plot the p values in order
%also plot the BH critical boundary
fig = LatexFigure.sub();
z_tester.plotPValues();
saveas(fig,fullfile('reports','figures','inference','empirical_null_sub_z_p_values.eps'),'epsc');