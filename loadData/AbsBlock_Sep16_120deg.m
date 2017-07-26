classdef AbsBlock_Sep16_120deg < Scan

    properties
    end
    
    methods
        
        function this = AbsBlock_Sep16_120deg()
            this@Scan('data/absBlock_CuFilter_Sep16/scans/phantom_120deg/', 'block120deg_', 2000, 2000, 20);
            
            this.addARTistFile('data/absBlock_CuFilter_Sep16/sim/phantom/sim120.tif');
            
            this.reference_scan_array = ReferenceArrayGetter.getReferenceScanArray_Sep16();
        end
        
    end
    
end

