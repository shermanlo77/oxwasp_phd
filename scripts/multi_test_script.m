%MULTIPLE HYPOTHESIS TEST SCRIPT
%Plots the histogram of simulated z statistics along with critical boundaries for various corrections
%Also plots the ordered p values for the BH procedure
%
%The following scenarios are investigated 
    %1000 N(0,1)
    %800 N(0,1) and 200 N(0,1)

clc;
close all;
clearvars;

%%%%%CASE 1%%%%%

%declare rng
rand_stream = RandStream('mt19937ar','Seed',uint32(3499211588));
%number of test statistics to be simulated
n = 1000;

%simulated all null
X = rand_stream.randn(n,1);

z_uncorrected = 2; %set sigma level
alpha = 2*(1-normcdf(z_uncorrected)); %significance level
z_bonferroni = -norminv(alpha/(2*n)); %get bonferroni boundary

%plot histogram of test statistics
fig = LatexFigure.main();
histogram_custom(X);
ax = gca;
hold on;
%plot the critical boundary for the uncorrected case and bonferroni case
plot([-z_uncorrected,-z_uncorrected],ax.YLim,'k-');
plot([-z_bonferroni,-z_bonferroni],ax.YLim,'b-.');
plot([z_uncorrected,z_uncorrected],ax.YLim,'k-');
plot([z_bonferroni,z_bonferroni],ax.YLim,'b-.');
xlabel('z');
ylabel('frequency density');
legend('z histogram','uncorrected boundary','bonferroni boundary');
saveas(fig,fullfile('reports','figures','inference','nullhisto.eps'),'epsc');

z_tester = ZTester(X);
z_tester.doTest();
fig = LatexFigure.main();
z_tester.plotPValues();
legend('p values','critical','Location','northwest');
saveas(fig,fullfile('reports','figures','inference','nullpvalues.eps'),'epsc');

%%%%%CASE 2%%%%%

%number of test statistics to be simulated
n = 1000;
%simulate true null test statistics
X = rand_stream.randn(n*4/5,1);
%simulate true alt test statistics
X = [X;2+rand_stream.randn(n*1/5,1)];

%instantise ZTester object and do the BH procedure
z_tester = ZTester(X);
z_tester.doTest();

z_uncorrected = 2; %set sigma level
alpha = 2*(1-normcdf(z_uncorrected)); %significance level
z_bonferroni = -norminv(alpha/(2*n)); %get bonferroni boundary
%get the boundary from the BH procedure
z_bh = z_tester.getZCritical();
z_bh = z_bh(end);
%save the boundary
file_id = fopen(fullfile('reports','figures','inference','alt_z_critical.txt'),'w');
fprintf(file_id,'%.2f',z_bh);
fclose(file_id);
%save the number of positive results for uncorrected
file_id = fopen(fullfile('reports','figures','inference','alt_n_positive_uncorrected.txt'),'w');
fprintf(file_id,'%.0f',sum(abs(X)>=z_uncorrected));
fclose(file_id);
%save the number of positive results for BH
file_id = fopen(fullfile('reports','figures','inference','alt_n_positive_bh.txt'),'w');
fprintf(file_id,'%.0f',sum(abs(X)>=z_bh));
fclose(file_id);

%plot histogram of test statistics
fig = LatexFigure.main();
histogram_custom(X);
ax = gca;
hold on;
%plot the critical boundary for the uncorrected case and bonferroni case
plot([-z_uncorrected,-z_uncorrected],ax.YLim,'k-');
plot([-z_bonferroni,-z_bonferroni],ax.YLim,'b-.');
plot([-z_bh,-z_bh],ax.YLim,'r--');
plot([z_uncorrected,z_uncorrected],ax.YLim,'k-');
plot([z_bonferroni,z_bonferroni],ax.YLim,'b-.');
plot([z_bh,z_bh],ax.YLim,'r--');
xlabel('z');
ylabel('frequency density');
legend('z histogram','uncorrected boundary','bonferroni boundary','BH boundary','Location','northwest');
saveas(fig,fullfile('reports','figures','inference','althisto.eps'),'epsc');

%plot the p values in order
fig = LatexFigure.main();
z_tester.plotPValues();
legend('p values','critical','Location','northwest');
saveas(fig,fullfile('reports','figures','inference','altpvalues.eps'),'epsc');
