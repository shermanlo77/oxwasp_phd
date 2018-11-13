%CLASS: EXPERIMENT Z NULL
%Experiment for simulating n N(0,1) and storing the emperical null mean and var
%Different n and parzen kernel width are investigated
%Plotted are log MSE vs kernel width vs log n, MSE for the null mean and null var
classdef Experiment_ZNull < Experiment
    
    %MEMBER VARIABLES
    properties (SetAccess = private)
        
        rng; %random number generator
        n_array; %array of n (sample size)
        k_array; %array of kernel width
        n_repeat; %number of times to repeat the simulation for a given k and n
        
        %array to store the mode and half width estimation using the Parzen density estimation
            %dim 1: for each repeat
            %dim 2: for each kernel width
            %dim 3: for each n or each sample size
        mean_array;
        var_array;

        i_iteration; %number of iterations done
        n_iteration; %number of iterations in the whole experiments
        
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        function this = Experiment_ZNull()
            this@Experiment('ZNull');
        end
        
        %IMPLEMENTED: PRINT RESULTS
        function printResults(this)
            
            %meshgrid for n and k
            [logn_plot,k_plot] = meshgrid(log10(this.n_array),this.k_array);
            factor_array = [0.9, 1.06, 1.144]; %array of fudge factors
            
            %for the mode estimation, then half width estimation
            for i_array = 1:3

                %get the corresponding array
                if i_array == 1
                    array = (squeeze(median(this.mean_array.^2)));
                    z_label = 'median squared error';
                elseif i_array == 2
                    array = squeeze( median( (sqrt(this.var_array)-1).^2 ) );
                    z_label = 'median squared error';
                else
                    array = squeeze(median(sqrt(this.var_array)));
                    z_label = '\sigma_0 estimate';
                end

                %surf plot the error vs k and log n
                fig = LatexFigure.sub();
                surf(k_plot,logn_plot,array);
                %label axis
                xlabel('bandwidth');
                ylabel('log(n)');
                zlabel(z_label);
                hold on;
                
                %plot the path of the rule of thumb
                %logn_path is array of logn to evaluate the rule of thumb
                logn_path = interp1((1:numel(this.n_array))',log10(this.n_array),linspace(1,numel(this.n_array),1000*numel(this.n_array))');
                if i_array ~= 3
                    %for each fudge factor
                    for i = 1:numel(factor_array)
                        %for each logn_path, work out the rule of thumb k
                        k_path = factor_array(i)*((10.^logn_path).^(-1/5));
                        %then for each k and logn pair, interpolate to get the value of the array
                        path = interp2(logn_plot,k_plot,array,logn_path,k_path) + 0.01;
                        %plot the error along the rule of thumb
                        plot3(k_path,logn_path,path,'LineWidth',3');
                    end
                    ax = gca;
                    ax.Children(numel(factor_array)).Color = [0,1,1];
                end
                %set the axis and view angle
                xlim(this.k_array([1,numel(this.k_array)]));
                ylim(log10(this.n_array([1,numel(this.n_array)])));

                %for the 3rd array
                if i_array==3
                    hold on;
                    %meshplot the true value of the null variance
                    ax = surf(k_plot,logn_plot,ones(size(array)));
                    ax.FaceColor = [0.75,0.75,0.75];
                    %ax.FaceAlpha = 0;
                    view(-33,20);
                else
                    view(164,44);
                    %plot legend
                    ax = gca;
                    legend(ax.Children([3,2,1]),{'0.9','1.06','1.144'},'Location','best');
                end
                
                %save the figure;
                saveas(fig,fullfile('reports','figures','inference',strcat(this.experiment_name,num2str(i_array),'.eps')),'epsc');

                %if this is a plot of the estimator itself, save the figure using a heatmap
                linestyle_array = {'k--','k-.','k:'};
                if i_array==3
                    fig = LatexFigure.sub();
                    %heatmap the array
                    imagesc(log10(this.n_array),this.k_array,array);
                    axis xy;
                    hold on;
                    %make a contour plot
                    contour(log10(this.n_array),this.k_array,array,[0,1],'k-');
                    %make a blank plot for the purpose of legend
                    plot([0,0],[0,0],'k-');
                    %for each factor
                    for i = 1:numel(factor_array)
                        %plot the rule of thumb
                        k_path = factor_array(i)*((10.^logn_path).^(-1/5));
                        plot(logn_path,k_path,linestyle_array{i},'LineWidth',1');
                    end
                    %label axis
                    ax = gca;
                    legend(ax.Children([4,3,2,1]),{'empirical truth','0.9','1.06','1.144'},'Location','northeast');
                    ylabel('bandwidth');
                    xlabel('log(n)');
                    colorbar;
                    saveas(fig,fullfile('reports','figures','inference',strcat(this.experiment_name,num2str(i_array+1),'.png')),'png');
                end
                
            end
        end
        
    end
    
    %PROTECTED METHODS
    methods (Access = protected)
        
        %IMPLEMENTED: SETUP
        function setup(this)
            %set random seed
            this.rng = RandStream('mt19937ar','Seed',uint32(2055696007));
            %declare arrays to store values to be investigated
            this.n_array = round(10.^linspace(1,6,40))';
            this.k_array = linspace(0.09,1.5,80)';
            %number of times to repeat the experiment
            this.n_repeat = 50;
            %declare array to store the mode and half width estimation using the Parzen density estimation
                %dim 1: for each repeat
                %dim 2: for each kernel width
                %dim 3: for each n or each sample size
            this.mean_array = zeros(this.n_repeat, numel(this.k_array), numel(this.n_array) );
            this.var_array = zeros(this.n_repeat, numel(this.k_array), numel(this.n_array) );
            %assign other member variables
            this.i_iteration = 0;
            this.n_iteration = this.n_repeat * (numel(this.n_array)) * (numel(this.k_array));
        end
        
        %IMPLEMENTED: DO EXPERIMENT
        function doExperiment(this)
            
            %for every n in n_array
            for i_n = 1:numel(this.n_array)

                %get n or sample size
                n = this.n_array(i_n);

                %for every kernel width
                for i_k = 1:numel(this.k_array)

                    %get the kernel width
                    k = this.k_array(i_k);

                    %for n_repeat times
                    for i_repeat = 1:this.n_repeat
                        
                        %simulate n N(0,1)
                        X = this.rng.randn(n,1);

                        %instantise z tester
                        z_tester = ZTester(X);
                        %set the kernel width
                        z_tester.setDensityEstimationParameter(k);
                        %get the mode and half width estimation
                        z_tester.estimateNull();

                        %save the mode and half width estimation
                        this.mean_array(i_repeat, i_k, i_n) = z_tester.mean_null;
                        this.var_array(i_repeat, i_k, i_n) = z_tester.var_null;           

                        %update iteartion and progress bar
                        this.i_iteration = this.i_iteration + 1;
                        this.printProgress(this.i_iteration / this.n_iteration);
                    end

                end

            end
            
            
        end

        
    end
    
end

