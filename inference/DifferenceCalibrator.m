%CLASS DIFFERENCE CALIBRATOR
%Used for calibrating the greyvalues in the aRTist image, calibrated using a held out scan
%A kernel regression is fitted onto (scan - aRTist) vs aRTist greyvalues
%The fitted regression is then used to correct the greyvalues in aRTist
%
%How to use:
%Pass a scan and aRTist image into the constructor, they must be in vector form (so that segmentation is done outside the class)
%Set the parameter of the kernel smoother using the method setParameter()
%Call the method calibrate()
%eExtract the corrected aRTist greyvalues from the member variable aRTist_calibrated
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
            %scan_vector: column vector of a scan, used for calibration
            %aRTist_vector: column vector of aRTist simulation, uncalibrated
        function this = DifferenceCalibrator(scan_vector, aRTist_vector)
            %assign member variables
            this.scan = scan_vector;
            this.aRTist = aRTist_vector;
        end
        
        %METHOD: SET PARAMETER:
        %Set parameter used in the kernel smoothing
        %PARAMETERS:
            %parameter: parameter used in kernel smoothing
        function setParameter(this, parameter)
            this.parameter = parameter;
        end
        
        %METHOD: CALIBRATE
        %Fit a kernel regression on (scan - aRTist) vs aRTist
        %Correct the aRTist greyvalues and save it to the member variable aRTist_calibrated
        function calibrate(this)
            %get the difference between the scan and aRTist
            d = this.scan - this.aRTist;
            %instantise a kernel smoother
            this.kernel_smoother = MeanVar_kNN(this.parameter);
            %train the kernel smoother
            this.kernel_smoother.train(this.aRTist,d);
            %predict the difference, given the aRTist image
            d_predict = this.kernel_smoother.predict(this.aRTist);
            %correct the aRTist greyvalue and save it to aRTist_calibrated
            this.aRTist_calibrated = this.aRTist + d_predict;
        end
        
        %METHOD: PLOT CALIBRATION
        %Plot aRTist bs scam
        %PARAMETERS:
            %hist_bin: 2 vector, parameter for the hist3Heatmap function
            %is_calibrated: true if to use the calibrated aRTist
        %RETURN:
            %fig: figure
        function fig = plotCalibration(this, hist_bin, is_calibrated)
            %if want the calibrated aRTist vector
            if is_calibrated
                %get the calibrated aRTist vector from the member variable
                aRTist_plot = this.aRTist_calibrated;
            %else, get the uncalibrated aRTist vector
            else
                aRTist_plot = this.aRTist;
            end
            
            %plot heatmap of aRTist vs scan
            fig = figure;
            hist3Heatmap(this.scan,aRTist_plot,hist_bin,true);
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
        
        %METHOD: PLOT DIFFERENCE
        %Based on the Bland-Altman plot
        %Plot scan - aRTist vs aRTist
        %PARAMETERS:
            %hist_bin: 2 vector, parameter for the hist3Heatmap function
            %is_calibrated: true if to use the calibrated aRTist
            %is_include_smoother: true if to include the kernel regression of (scan - uncalibrated aRTist) vs uncalibrated aRTist
        %RETURN:
            %fig: figure
        function fig = plotDifference(this, hist_bin, is_calibrated, is_include_smoother)
            %if want the calibrated aRTist vector
            if is_calibrated
                aRTist_plot = this.aRTist_calibrated;
            else
                aRTist_plot = this.aRTist;
            end
            
            %get the difference betweenthe scan and aRTist
            d = this.scan - aRTist_plot;
            
            %plot heatmap
            fig = figure;
            hist3Heatmap(aRTist_plot,d,hist_bin,true);
            hold on;
            %label axis
            colorbar;
            xlabel('aRTist greyvalue (arb. unit)');
            ylabel('difference in greyvalue (arb. unit)');

            %if to include the kernel regression
            if is_include_smoother
                %get the range of aRTist greyvalues
                aRTist_plot = round(min(aRTist_plot)):round(max(aRTist_plot));
                plot(aRTist_plot,this.kernel_smoother.predict(aRTist_plot),'-r');
            end
            
        end
        
    end
    
end

