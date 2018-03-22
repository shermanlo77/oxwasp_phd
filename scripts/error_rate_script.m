%SCRIPT: ERROR RATE SCRIPT
%Various error rates are investigated (PCER, FWER, FDR)
%when using different types of corrections (no correction, Bonferroni, BH)
%for multiple hypothesis testing at the z_α = 2 level.
%1 000 test statistics were simulated
    %case 1: all N(0,1)
    %case 2: 800 N(0,1) 200 (2,1)
%This was repeated 1 000 repeats of the experiment
%Results are printed onto a latex table

clc;
clearvars;
close all;

n_repeat = 1000; %number of times to repeat the experiment
n = 1000; %number of test statistics
z_alpha = 2; %threshold
alpha = 2*(1-normcdf(z_alpha)); %significance level
rand_stream = RandStream('mt19937ar','Seed',uint32(1390624813));

%for each case, 1:2
for i_case = 1:2
    
    %get the proportion of null
    switch i_case
        case 1
            pi_0 = 1;
        case 2
            pi_0 = 4/5;
    end

    %declare arrays, storing counts of a specific error or test result
        %dim 1: for each repeat
        %dim 2: for (no correction, Bonferroni, BH)
    false_positive_array = zeros(n_repeat,3); %stores the number of false positive results
    n_positive_array = zeros(n_repeat,3); %stores the number of positive results

    %for n_repeat times
    for i = 1:n_repeat

        %simulate n test statistics
        Z = zeros(n,1);
        n_0 = round(pi_0*n); %get the number of true null
        %simulate true null
        Z(1:n_0) = rand_stream.randn(n_0,1);
        %simulate non-true null (if any)
        Z((n_0+1):end) = 2+rand_stream.randn(n-n_0,1);

        %declare boolean array is_null
        %true of a test statistic is truly null
        is_null = zeros(n,1);
        is_null(1:n_0) = true;

        %%%UNCORRECTED hypothesis testing
        is_positive = abs(Z)>z_alpha; %get boolean, is it a positive result
        n_positive_array(i,1) = sum(is_positive); %get the number of positive results
        false_positive_array(i,1) = sum(is_positive & is_null); %get the number of false positive results

        %BONFERONNI hypothesis testing
        is_positive = abs(Z)>norminv(1-alpha/(z_alpha*n)); %get boolean, is it a positive result
        n_positive_array(i,2) = sum(is_positive); %get the number of positive results
        false_positive_array(i,2) = sum(is_positive & is_null); %get the number of false positive results

        %BH hypothesis testing
        %instantise a z tester and do the test
        z_tester = ZTester(Z);
        z_tester.setSigma(z_alpha);
        z_tester.doTest();
        is_positive = z_tester.sig_image; %get boolean, is it a positive result
        n_positive_array(i,3) = sum(is_positive); %get the number of positive results
        false_positive_array(i,3) = sum(is_positive & is_null); %get the number of false positive results
    end

    %get array of FDR (one for each repeat)
    fdr = false_positive_array ./ n_positive_array;
    fdr(isnan(fdr)) = 0;

    %get array of PCER (one for each repeat)
    pcer = false_positive_array / n;

    %work out FWER
    %get boolean, any repeats which has at least one false positive
    has_false_positive = false_positive_array > 0;
    %get the posterior parameters for FWER estimate
    beta_a = sum(has_false_positive,1);
    beta_b = sum(~has_false_positive,1);
    %work out the estimate of the FWER
    fwer = (beta_a)./(beta_a+beta_b);
    %work out the estimated standard error of FWER
    fwer_err = sqrt( (beta_a.*beta_b) ./ ( ((beta_a+beta_b).^2) .* (beta_a+beta_b+1) ) );

    %declare array for storing the mean and standard error in a table
        %dim 1: for each error rate (PCER, FWER, FDR)
        %dim 2: for each (no correction, Bonferroni, BH)
    mean_value = zeros(3,3);
    err = zeros(3,3);

    %for each row, work out the error rates
    mean_value(1,:) = mean(pcer,1); %PCER
    mean_value(2,:) = fwer; %FWER
    mean_value(3,:) = mean(fdr,1); %FDR

    %for each row, work out the standard errors
    err(1,:) = std(pcer,[],1)/sqrt(n_repeat); %PCER using normal approximation
    err(2,:) = fwer_err; %FWER using bayesian
    err(3,:) = std(fdr,[],1)/sqrt(n_repeat); %FDR using normal approximation

    %output the table to a latex table
    latex_table = LatexTable(mean_value, err, {'PCER','FWER','FDR'}, {'No correction','Bonferroni','BH'});
    latex_table.setNDecimalForNoError(1); %there are 0 errors in some cases, set the number of decimial place to be 0 in that case
    %output the latex table
    latex_table.print(fullfile('reports','tables',strcat('inference_error_rate',num2str(i_case),'.tex_table')));
end