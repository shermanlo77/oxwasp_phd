classdef Experiment_MeanVarFit_Sep16_120deg < Experiment_MeanVarFit_Sep16
    
    properties
    end
    
    methods (Access = public)
        
        %CONSTRUCTOR
        function this = Experiment_MeanVarFit_Sep16_120deg()
            %call superclass with experiment name
            this@Experiment_MeanVarFit_Sep16('MeanVarFit_Sep16_120deg');
        end
        
    end
    
    methods (Access = protected)
        
        %IMPLEMENTED: GET SCAN
        function scan = getScan(this)
            scan = AbsBlock_Sep16_120deg();
        end
        
    end
    
end

