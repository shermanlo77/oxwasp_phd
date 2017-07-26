classdef AbsBlock_Sep16_30deg < Scan

    properties
    end
    
    methods
        
        function this = AbsBlock_Sep16_30deg()
            this@Scan('data/absBlock_CuFilter_Sep16/scans/phantom_30deg/', 'block30deg_', 2000, 2000, 20);
            
            this.addARTistFile('data/absBlock_CuFilter_Sep16/sim/phantom/sim30.tif');
            
            this.reference_scan_array = ReferenceArrayGetter.getReferenceScanArray_Sep16();
        end
        
    end
    
end

