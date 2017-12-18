classdef DifferenceCalibrator < handle
    
    %MEMBER VARIABLES
    properties (SetAccess = private)
        scan; %column vector of a scan, used for calibration
        aRTist; %column vector of aRTist, in the uncalibrated state
        aRTist_calibrated; %column vector of aRTist, in the calibrated state
        parameter; %parameter in kernel smoothing
        kernel_smoother; %the trained kernel smoother
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        %PARAMETERS:
            %scan: column vector of a scan, used for calibration
            %aRTist: column vector of aRTist simulation, uncalibrated
        function this = DifferenceCalibrator(scan, aRTist)
            %assign member variables
            this.scan = scan;
            this.aRTist = aRTist;
        end
        
        %METHOD: SET PARAMETER:
        %Set parameter used in the kernel smoothing
        %PARAMETERS:
            %parameter: parameter used in kernel smoothing
        function setParameter(this, parameter)
            this.parameter = parameter;
        end
        
        function calibrate(this)
            d = this.scan - this.aRTist;
            this.kernel_smoother = MeanVar_kNN(this.parameter);
            this.kernel_smoother.train(this.aRTist,d);
            d_predict = this.kernel_smoother.predict(this.aRTist);
            this.aRTist_calibrated = this.aRTist + d_predict;
        end
        
        function fig = plotCalibration(this,hist_bin)
            fig = figure;
            hist3Heatmap(this.scan,this.aRTist,hist_bin,true);
            hold on;
            %get the min and max greyvalue
            min_grey = min([min(this.scan),min(this.aRTist)]);
            max_grey = max([max(this.scan),max(this.aRTist)]);
            %plot straight line with gradient 1
            plot([min_grey,max_grey],[min_grey,max_grey],'r');
            %label axis
            colorbar;
            xlabel('phantom greyvalue (arb. unit)');
            ylabel('aRTist greyvalue (arb. unit)');
        end
        
        function fig = plotDifference(this,hist_bin, is_calibrated, is_include_smoother)
            
            if is_calibrated
                aRTist_plot = this.aRTist_calibrated;
            else
                aRTist_plot = this.aRTist;
            end
            
            d = this.scan - aRTist_plot;
            
            figure;
            hist3Heatmap(aRTist_plot,d,hist_bin,true);
            hold on;
            %label axis
            colorbar;
            xlabel('aRTist greyvalue (arb. unit)');
            ylabel('difference in greyvalue (arb. unit)');

            if is_include_smoother
                plot(aRTist_plot,this.kernel_smoother.predict(aRTist_plot),'-r');
            end
            
        end
        
    end
    
end

