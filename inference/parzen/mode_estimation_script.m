clc;
close all;
clearvars;

rng(uint32(2055696007), 'twister');

n_array = round(10.^linspace(1,5,20))';
k_array = linspace(0.09,2.2,20)';
n_repeat = 50;

mean_array = zeros(n_repeat, numel(k_array), numel(n_array) );
std_array = zeros(n_repeat, numel(k_array), numel(n_array) );

for i_n = 1:numel(n_array)
    
    n = n_array(i_n);
    
    for i_k = 1:numel(k_array)
        
        k = k_array(i_k);
        
        for i_repeat = 1:n_repeat
            
            X = normrnd(0,1,n,1);
            density_estimator = Parzen(X);
            density_estimator.setParameter(k * min([std(X),iqr(X)/1.34]) );
            [mean_null, std_null] = density_estimator.estimateNull(100);
            
            mean_array(i_repeat, i_k, i_n) = mean_null;
            std_array(i_repeat, i_k, i_n) = std_null;
            
        end
        
    end
    
end

mean_array_plot = squeeze(nanmean(mean_array.^2));
std_array_plot = squeeze(nanmean((std_array-1).^2));

[n_plot,k_plot] = meshgrid(log10(n_array),k_array);
path = n_array.^(-1/5);
factor_array = [0.9, 1.144, 2, 3.33];

for i_array = 1:2
    
    if i_array == 1
        array = mean_array_plot;
    else
        array = std_array_plot;
    end
    
    figure;
    surf(k_plot,n_plot,array);
    xlabel('Parzen std');
    ylabel('log(n)');
    zlabel('Mean squared error');
    hold on;
    for i = 1:numel(factor_array)
        k_path = factor_array(i) * path;
        error_path = zeros(numel(n_array),1);
        for j = 1:numel(n_array)
            k = k_path(j);
            [~,k_index] = sort(abs(k-k_array));

            if k_index(1) > k_index(2)
                k_index(1:2) = flipud(k_index(1:2));
            end
            k_neighbour = k_array(k_index(1:2));

            r = (k - k_neighbour(1)) / (k_neighbour(2) - k_neighbour(1));
            error_path(j) = r*(array(k_index(2),j) - array(k_index(1),j)) + array(k_index(1),j);
        end
        hold on;
        ax = plot3(k_path,log10(n_array),error_path,'LineWidth',2');
    end
    xlim(k_array([1,numel(k_array)]));
    ylim(log10(n_array([1,numel(n_array)])));
    view(-166,34);
    
    ax = gca;
    legend(ax.Children([4,3,2,1]),{'0.9','1.144','2','3.33'},'Location','best');
end