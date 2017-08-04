%EXPERIMENT GLM VAR MEAN MAR 16
%See superclass Experiment_GLMVarMean
%For the dataset AbsBlock_Mar13, only top half of the image is used to avoid the foam
classdef Experiment_GLMVarMean_Mar16 < Experiment_GLMVarMean
    
    properties
    end
    
    methods
        
        %CONSTRUCTOR
        function this = Experiment_GLMVarMean_Mar16()
            %call superclass with experiment name
            this@Experiment_GLMVarMean('GLMVarMean_Mar16');
        end
        
        %OVERRIDE: SET UP EXPERIMENT
        function setUpExperiment(this)
            %call superclass with 100 repeats and a random seed
            this.setUpExperiment@Experiment_GLMVarMean(100, uint32(176048084));
        end
        
        %OVERRIDE: SAVE SEGMENTATION
        %For this class, only conisder the top half
        function saveSegmentation(this, segmentation)
            [height,~] = size(segmentation);
            this.segmentation = reshape(segmentation(1:(height/2),:),[],1);
        end
        
        %OVERRIDE: GET MEAN VARIANCE
        %For this class, only conisder the top half
        function [sample_mean,sample_var] = getMeanVar(this, data_index)
            scan = this.getScan();
            [sample_mean,sample_var] = scan.getSampleMeanVar_topHalf(data_index);
            %segment the mean var data
            sample_mean = sample_mean(this.segmentation);
            sample_var = sample_var(this.segmentation);
        end
        
        %IMPLEMENTED: GET SCAN
        function scan = getScan(this)
            scan = AbsBlock_Mar16();
        end
        
        %IMPLEMENTED: GET N GLM
        function n_glm = getNGlm(this)
            n_glm = 9;
        end
        
        %IMPLEMENTED: GET GLM
        function model = getGlm(this, shape_parameter, index)
            switch index
                case 1
                    model = MeanVar_GLM_identity(shape_parameter,1);
                case 2
                    model = MeanVar_GLM_canonical(shape_parameter,-1);
                case 3
                    model = MeanVar_GLM_canonical(shape_parameter,-2);
                case 4
                    model = MeanVar_GLM_canonical(shape_parameter,-3);
                case 5
                    model = MeanVar_GLM_canonical(shape_parameter,-4);
                case 6
                    model = MeanVar_GLM_log(shape_parameter,-1);
                case 7
                    model = MeanVar_GLM_log(shape_parameter,-2);
                case 8
                    model = MeanVar_GLM_log(shape_parameter,-3);
                case 9
                    model = MeanVar_GLM_log(shape_parameter,-4);
            end
        end
        
        %IMPLEMENTED: GET N BIN
        function n_bin = getNBin(this)
            n_bin = 100;
        end
        
    end
    
end

