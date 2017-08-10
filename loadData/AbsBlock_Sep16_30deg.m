classdef AbsBlock_Sep16_30deg < AbsBlock_Sep16

    properties
    end
    
    methods
        
        function this = AbsBlock_Sep16_30deg()
            this@AbsBlock_Sep16('data/absBlock_CuFilter_Sep16/scans/phantom_30deg/', 'block30deg_');
            this.addARTistFile('data/absBlock_CuFilter_Sep16/sim/phantom/sim30.tif');
        end
        
    end
    
end

