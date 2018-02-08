%CLASS: EXPERIMENT SIMULATE ROC
%See superclass Experiment_NoDefect
%
%This class does an experiment simulating defects of different intensities
%ROC (Receiver operating characteristic) curves are plotted for different defect intensities
%ROC curves are repeated multiple times
%Ares of the ROC curves are calculated and plotted vs defect intensity
classdef Experiment_SimulateRoc < Experiment_NoDefect
    
    properties
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        function this = Experiment_SimulateRoc(name)
            this@Experiment_NoDefect(name);
        end
        
        %OVERRIDE: PRINT RESULTS
        function printResults(this)
            
            roc_area = zeros(this.n_repeat,numel(this.parameter_array));
            
            %for each parameter
            for i_parameter = 1:numel(this.parameter_array)
                figure;
                ax = gca;
                %for each repeat
                for i_repeat = 1:this.n_repeat
                    %plot the ROC for this repeat
                    index = ((i_repeat-1)*numel(this.size_array)+1) : (i_repeat*numel(this.size_array));
                    x = this.fdr_array(index, 1, i_parameter );
                    y = this.fdr_array(index, 2, i_parameter );
                    plot(x,y,'Color',ax.ColorOrder(1,:));
                    hold on;
                    %get the area of the roc
                    trapezium_areas = 0.5*(y(1:(end-1))+y(2:end)).*(x(2:end)-x(1:(end-1)));
                    roc_area(i_repeat,i_parameter) = sum(trapezium_areas);
                end
                plot([0,1],[0,1],'k--');
                xlabel('false positive rate');
                ylabel('true positive rate');
                %scatter plot all TPR vs FPR points
                %scatter(this.fdr_array(:,1,i_parameter),this.fdr_array(:,2,i_parameter),'MarkerEdgeColor',ax.ColorOrder(2,:),'Marker','x');
            end
            
            %boxplot the area of roc vs defect intensity
            figure;
            box_plot = Boxplots(roc_area,true);
            box_plot.setPosition(this.parameter_array);
            box_plot.plot();
            xlabel('defect intensity');
            ylabel('area of roc');
            
            %plot aRTist and the result of the test of one specific saved example
            this.printConvolution();
        end
        
    end
    
    %PROTECTED METHODS
    methods (Access = protected)
        
        %OVERRIDE: SETUP EXPERIMENT
        %PARAMETERS:
            %rng: random number generator
            %parameter_array: array of parameters for the function
        function setup(this, rng)
            this.rng = rng;
            this.parameter_array = 10.^linspace(2,3,20);
            this.size_array = linspace(0,1,41);
            this.n_repeat = 5;
            this.fdr_array = zeros(this.n_repeat*numel(this.size_array), 2, numel(this.parameter_array));
            this.plot_index = [3;round(numel(this.parameter_array)/2)];
            this.i_iteration = 0;
            this.n_iteration = numel(this.parameter_array) * this.n_repeat;
        end
        
        %METHOD: GET FDR
        %Given a convolution, do the test, get the false positive rate and save it
        %PARAMETERS:
            %convolution: EmpericalConvolution object
            %defect_simulator: the defect simulator for this current iteration
            %i_parameter: iteration integer
            %i_repeat: interation integer
            %i_size: iteration integer
        function getFdr(this, convolution, defect_simulator, i_parameter, i_repeat, i_size)
            %set the threshold of the test and do the test
            convolution.setSize(this.size_array(i_size));
            convolution.doTest();
            %get the false positive rate
            this.fdr_array( (i_repeat-1)*numel(this.size_array)+i_size, 1, i_parameter ) = sum(sum(convolution.sig_image & (~defect_simulator.sig_image)))/defect_simulator.n_null;
            %get the true positive rate
            this.fdr_array( (i_repeat-1)*numel(this.size_array)+i_size, 2, i_parameter ) = sum(sum(convolution.sig_image & (defect_simulator.sig_image)))/defect_simulator.n_sig;
        end
        
        %METHOD: GET DATA
        %Return an object containing images
        function data = getData(this)
            data = AbsBlock_July16_30deg();
            data.addDefaultShadingCorrector();
        end
        
    end
    
end

