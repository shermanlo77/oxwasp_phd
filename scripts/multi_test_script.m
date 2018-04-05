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

%instantise and set up various z testers
uncorrected_test = ZTester_Uncorrected(X);
bonferroni_test = ZTester_Bnfrrn(X);
z_tester = ZTester(X);
uncorrected_test.doTest();
bonferroni_test.doTest();
z_tester.doTest();
uncorrected_test.setCriticalColour([0,0.4470,0.7410]);
bonferroni_test.setCriticalColour([0.9290,0.6940,0.1250]);

%plot histogram of test statistics
fig = LatexFigure.main();
uncorrected_test.plotHistogram();
ax = gca;
ax.XLim = [-5,5];
hold on;
%plot the critical boundary for the uncorrected case and bonferroni case
uncorrected_test.plotCritical();
bonferroni_test.plotCritical();
xlabel('z');
ylabel('frequency density');
legend(ax.Children([5,4,2]),'z histogram','uncorrected boundary','bonferroni boundary');
saveas(fig,fullfile('reports','figures','inference','nullhisto.eps'),'epsc');

%plot the ordered p values
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

%instantise and set up various z testers
z_tester = ZTester(X);
z_tester.doTest();
bonferroni_test = ZTester_Bnfrrn(X);
bonferroni_test.doTest();
bonferroni_test.setCriticalColour([0.9290,0.6940,0.1250]);
uncorrected_test = ZTester_Uncorrected(X);
uncorrected_test.doTest();
uncorrected_test.setCriticalColour([0,0.4470,0.7410]);

%get the boundary from the BH procedure
z_bh = z_tester.getZCritical();
z_bh = z_bh(end);
%save the boundary
file_id = fopen(fullfile('reports','figures','inference','alt_z_critical.txt'),'w');
fprintf(file_id,'%.2f',z_bh);
fclose(file_id);
%save the number of positive results for uncorrected
file_id = fopen(fullfile('reports','figures','inference','alt_n_positive_uncorrected.txt'),'w');
fprintf(file_id,'%.0f',sum(uncorrected_test.sig_image));
fclose(file_id);
%save the number of positive results for BH
file_id = fopen(fullfile('reports','figures','inference','alt_n_positive_bh.txt'),'w');
fprintf(file_id,'%.0f',sum(z_tester.sig_image));
fclose(file_id);

%plot histogram of test statistics
fig = LatexFigure.main();
z_tester.plotHistogram();
ax = gca;
ax.XLim = [-4,5];
hold on;
%plot the critical boundary
uncorrected_test.plotCritical();
bonferroni_test.plotCritical();
z_tester.plotCritical();
xlabel('z');
ylabel('frequency density');
legend(ax.Children([6,5,3,1]),'z histogram','uncorrected boundary','bonferroni boundary','BH boundary','Location','northwest');
saveas(fig,fullfile('reports','figures','inference','althisto.eps'),'epsc');

%plot the p values in order
fig = LatexFigure.main();
z_tester.plotPValues();
legend('p values','critical','Location','northwest');
saveas(fig,fullfile('reports','figures','inference','altpvalues.eps'),'epsc');
