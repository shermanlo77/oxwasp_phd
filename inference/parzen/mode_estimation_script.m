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
n_array = round(10.^linspace(1,6,20))'; %array of n (sample size)
k_array = linspace(0.09,1.5,20)'; %array of kernel width
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
mean_array_plot = log10(squeeze(nanmedian(mean_array.^2)));
std_array_plot = log10(squeeze(nanmedian((std_array-1).^2)));

%meshgrid for n and k
[n_plot,k_plot] = meshgrid(log10(n_array),k_array);

%declare rule of thumb curve
path = n_array.^(-1/5);
factor_array = [0.9866, 1.144]; %array of fudge factors


k_array_plot = linspace(min(k_array),max(k_array),10*numel(k_array))'; %array of kernel width
error_plot = k_array;
n_bootstrap = 100;
k_optima = zeros(n_bootstrap,numel(n_array));
for i_n = 1:numel(n_array)
    
    array = log((std_array(:,:,i_n)-1).^2);
    
    figure;
    box_plot = Boxplots(array,true);
    box_plot.setPosition(k_array);
    box_plot.plot();
    hold on;
    fitter = LocalLinearRegression(repmat(k_array,n_repeat,1), reshape(array',[],1));
    for i_k = 1:numel(k_array_plot)
        y_0 = fitter.getRegression(k_array_plot(i_k));
        error_plot(i_k) = y_0;
    end
    plot(k_array_plot,error_plot);
    
    [~,i_k_optima] = min(error_plot);
    k_optima(1,i_n) = k_array_plot(i_k_optima);
    
    array_bootstrap = array;
    for i_bootstrap = 2:n_bootstrap
        
        for i_k = 1:numel(k_array)
            array_bootstrap(:,i_k) = array(randi([1,n_repeat],n_repeat,1),i_k);
        end
        
        fitter = LocalLinearRegression(repmat(k_array,n_repeat,1), reshape(array_bootstrap',[],1));
        for i_k = 1:numel(k_array_plot)
            y_0 = fitter.getRegression(k_array_plot(i_k));
            error_plot(i_k) = y_0;
        end

        [~,i_k_optima] = min(error_plot);
        k_optima(i_bootstrap,i_n) = k_array_plot(i_k_optima);
    end
end

y = reshape(k_optima',[],1);
y_scale = std(y);
y = y/y_scale;
X = [ones(n_bootstrap*numel(n_array),1),repmat(n_array.^(-1/5),n_bootstrap,1)];
x_shift = mean(X(:,2));
x_scale = std(X(:,2));
X(:,2) = (X(:,2)-x_shift)/x_scale;
[~,~,stats] = glmfit(X,y,'gamma','link','identity','constant','off');
y_scale * stats.beta(2)/x_scale
sqrt(stats.covb(end))*y_scale/x_scale
y_scale * (stats.beta(1) - stats.beta(2)*x_shift/x_scale)
sqrt(y_scale^2*stats.covb(1)+(x_shift*y_scale/x_scale)^2*stats.covb(end) + 2*y_scale*(x_shift*y_scale/x_scale)*stats.covb(2))
%model_fit = fitglm(x,y,'Link','log','Distribution','normal');

boxplot_k_optima = Boxplots(k_optima,true);
boxplot_k_optima.setPosition((n_array).^(-1/5));
figure;
boxplot_k_optima.plot();




%for the mode estimation, then half width estimation
for i_array = 1:3
    
    %get the corresponding array
    if i_array == 1
        array = mean_array_plot;
        z_label = 'log MSE';
    elseif i_array == 2
        array = std_array_plot;
        z_label = 'log MSE';
    else
        array = squeeze(nanmean(std_array));
        z_label = 'H0 std estimate';
    end
    
    %surf plot the error vs 
    figure;
    surf(k_plot,n_plot,array);
    %label axis
    xlabel('kernel width');
    ylabel('log(n)');
    zlabel(z_label);
    hold on;
    %for each fudge factor
    for i = 1:numel(factor_array)
        %get the rule of thumb kernel width for each n
        k_path = factor_array(i) * path + 0.1739;
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
    legend(ax.Children([2,1]),{'0.9','1.144'},'Location','best');
    
    if i_array==3
        hold on;
        ax = mesh(k_plot,n_plot,ones(size(array)));
        ax.FaceAlpha = 0;
    end
end