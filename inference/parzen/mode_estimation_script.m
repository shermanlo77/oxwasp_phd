%SCRIPT: MODE ESTIMATION EXPERIMENT
%Assess the performance of emperical null estimation using the Parzen density estimation
%The mode and half width of the null density is estimated, the error in the estimation is investigated
%Different sample size and kernel width are investigated, and repeated
%
%Plots error vs kernel width and log(n)
%Error for mode estimation and half width estimation

clc;
close all;
clearvars;

%set random seed
rng(uint32(2055696007), 'twister');

%declare arrays to store values to be investigated
n_array = round(10.^linspace(1,5,20))'; %array of n (sample size)
k_array = linspace(0.09,2.2,20)'; %array of kernel width
%number of times to repeat the experiment
n_repeat = 50;

%declare array to store the mode and half width estimation using the Parzen density estimation
    %dim 1: for each repeat
    %dim 2: for each kernel width
    %dim 3: for each n or each sample size
mean_array = zeros(n_repeat, numel(k_array), numel(n_array) );
std_array = zeros(n_repeat, numel(k_array), numel(n_array) );

%for every n in n_array
for i_n = 1:numel(n_array)
    
    %get n or sample size
    n = n_array(i_n);
    
    %for every kernel width
    for i_k = 1:numel(k_array)
        
        %get the kernel width
        k = k_array(i_k);
        
        %for n_repeat times
        for i_repeat = 1:n_repeat
            
            %simulate n N(0,1)
            X = normrnd(0,1,n,1);
            
            %instantise z tester
            z_tester = ZTester(X);
            %set the kernel width
            z_tester.setDensityEstimationParameter(k * min([std(X),iqr(X)/1.34]) );
            %get the mode and half width estimation
            z_tester.estimateNull(100);
            
            %save the mode and half width estimation
            mean_array(i_repeat, i_k, i_n) = z_tester.mean_null;
            std_array(i_repeat, i_k, i_n) = z_tester.std_null;
            
        end
        
    end
    
end

%take the mean squared error over repeats
mean_array_plot = squeeze(nanmean(mean_array.^2));
std_array_plot = squeeze(nanmean((std_array-1).^2));

%meshgrid for n and k
[n_plot,k_plot] = meshgrid(log10(n_array),k_array);

%declare rule of thumb curve
path = n_array.^(-1/5);
factor_array = [0.9, 1.144, 2, 3.33]; %array of fudge factors

%for the mode estimation, then half width estimation
for i_array = 1:2
    
    %get the corresponding array
    if i_array == 1
        array = mean_array_plot;
    else
        array = std_array_plot;
    end
    
    %surf plot the error vs 
    figure;
    surf(k_plot,n_plot,array);
    %label axis
    xlabel('Parzen std');
    ylabel('log(n)');
    zlabel('Mean squared error');
    hold on;
    %for each fudge factor
    for i = 1:numel(factor_array)
        %get the rule of thumb kernel width for each n
        k_path = factor_array(i) * path;
        %declare array of error along this path
        error_path = zeros(numel(n_array),1);
        %for each n
        for j = 1:numel(n_array)
            %get the kernel width using the rule of thumb
            k = k_path(j);
            %get the k which is closest to a k in k_array
            [~,k_index] = sort(abs(k-k_array));
            %order the k_index so that k_index(1) < k_index(2)
            if k_index(1) > k_index(2)
                k_index(1:2) = flipud(k_index(1:2));
            end
            %the 2 k neighbouring k_path(j) is k_neighbour
            k_neighbour = k_array(k_index(1:2));

            %interpolate error using the 2 neighbouring ks
            r = (k - k_neighbour(1)) / (k_neighbour(2) - k_neighbour(1));
            error_path(j) = r*(array(k_index(2),j) - array(k_index(1),j)) + array(k_index(1),j);
        end
        hold on;
        %plot the error along the rule of thumb
        plot3(k_path,log10(n_array),error_path,'LineWidth',2');
    end
    %set the axis and view angle
    xlim(k_array([1,numel(k_array)]));
    ylim(log10(n_array([1,numel(n_array)])));
    view(-166,34);
    ax = gca;
    legend(ax.Children([4,3,2,1]),{'0.9','1.144','2','3.33'},'Location','best');
end