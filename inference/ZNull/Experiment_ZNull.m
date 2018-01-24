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
        
        %array for number of fails when estimating the null
            %dim 1: for each kernel width
            %dim 2: for each n
        fail_count_array;
        
        null_linspace; %number of points to evaluate to search for emperical mean
        
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
            
            %declare rule of thumb curve
            path = this.n_array.^(-1/5);
            %meshgrid for n and k
            [n_plot,k_plot] = meshgrid(log10(this.n_array),this.k_array);
            factor_array = [0.9866, 1.144]; %array of fudge factors
            
            %for the mode estimation, then half width estimation
            for i_array = 1:3

                %get the corresponding array
                if i_array == 1
                    array = log10(squeeze(median(this.mean_array.^2)));
                    z_label = 'log MSE';
                elseif i_array == 2
                    array = log10(squeeze(median((this.var_array-1).^2)));
                    z_label = 'log MSE';
                else
                    array = squeeze(median(this.var_array));
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
                    k_path = factor_array(i) * path;
                    %declare array of error along this path
                    error_path = zeros(numel(this.n_array),1);
                    %for each n
                    for j = 1:numel(this.n_array)
                        %get the kernel width using the rule of thumb
                        k = k_path(j);
                        %get the k which is closest to a k in k_array
                        [~,k_index] = sort(abs(k-this.k_array));
                        %order the k_index so that k_index(1) < k_index(2)
                        if k_index(1) > k_index(2)
                            k_index(1:2) = flipud(k_index(1:2));
                        end
                        %the 2 k neighbouring k_path(j) is k_neighbour
                        k_neighbour = this.k_array(k_index(1:2));

                        %interpolate error using the 2 neighbouring ks
                        r = (k - k_neighbour(1)) / (k_neighbour(2) - k_neighbour(1));
                        error_path(j) = r*(array(k_index(2),j) - array(k_index(1),j)) + array(k_index(1),j);
                    end
                    hold on;
                    %plot the error along the rule of thumb
                    plot3(k_path,log10(this.n_array),error_path,'LineWidth',2');
                end
                %set the axis and view angle
                xlim(this.k_array([1,numel(this.k_array)]));
                ylim(log10(this.n_array([1,numel(this.n_array)])));
                view(-166,34);
                ax = gca;
                legend(ax.Children([2,1]),{'0.9','1.144'},'Location','best');

                if i_array==3
                    hold on;
                    ax = mesh(k_plot,n_plot,ones(size(array)));
                    ax.FaceAlpha = 0;
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
            this.n_array = round(10.^linspace(1,6,20))';
            this.k_array = linspace(0.09,1.5,20)';
            %number of times to repeat the experiment
            this.n_repeat = 50;
            %declare array to store the mode and half width estimation using the Parzen density estimation
                %dim 1: for each repeat
                %dim 2: for each kernel width
                %dim 3: for each n or each sample size
            this.mean_array = zeros(this.n_repeat, numel(this.k_array), numel(this.n_array) );
            this.var_array = zeros(this.n_repeat, numel(this.k_array), numel(this.n_array) );
            %declare array for number of fails when estimating the null
                %dim 1: for each kernel width
                %dim 2: for each n
            this.fail_count_array = zeros(numel(this.k_array), numel(this.n_array) );
            %assign other member variables
            this.null_linspace = 500;
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
                        
                        %set a boolean, becomes true when null estimation is successful
                        got_null = false;

                        %while a null hasn't been found
                        while ~got_null
                            %simulate n N(0,1)
                            X = this.rng.randn(n,1);

                            %instantise z tester
                            z_tester = ZTester(X);
                            %set the kernel width
                            z_tester.setDensityEstimationParameter(k);
                            %get the mode and half width estimation
                            z_tester.estimateNull(this.null_linspace);
                            
                            %if the null std is not nan, set the flag got_null to true
                            if ~isnan(z_tester.std_null)
                                got_null = true;
                            %else, the null estimation failed, update fail_count_array
                            else
                                this.fail_count_array(i_k, i_n) = this.fail_count_array(i_k, i_n) + 1;
                            end
                        end

                        %save the mode and half width estimation
                        this.mean_array(i_repeat, i_k, i_n) = z_tester.mean_null;
                        this.var_array(i_repeat, i_k, i_n) = z_tester.std_null^2;

                        %update iteartion and progress bar
                        this.i_iteration = this.i_iteration + 1;
                        this.printProgress(this.i_iteration / this.n_iteration);
                    end

                end

            end
            
            
        end

        
    end
    
end

