%STATIC CLASS: REFERENCE ARRAY GETTER
%Contains functions which returns an array of Scan objects for each reference scan
%Each Scan object (except for the black image) has the associated aRTist image
classdef ReferenceArrayGetter
    
    properties
    end
    
    %STATIC METHODS
    methods (Static)
        
        %GET REFERENCE SCAN ARRAY July 16
        function [reference_scan_array,reference_white] = getReferenceScanArray_July16()
            reference_scan_array(6) = Scan();
            reference_scan_array(1) = Scan('data/absBlock_noFilter_July16/scans/blank_0W/', 'block_black_', 2000, 2000, 20, 0, 0, 708);
            reference_scan_array(2) = Scan('data/absBlock_noFilter_July16/scans/blank_10W/', 'block10w_', 2000, 2000, 20, 80, 10.08, 708);
            reference_scan_array(3) = Scan('data/absBlock_noFilter_July16/scans/blank_18W/', 'block18w_', 2000, 2000, 20, 80, 18.08, 708);
            reference_scan_array(4) = Scan('data/absBlock_noFilter_July16/scans/blank_28W/', 'block28w_', 2000, 2000, 20, 80, 28, 708);
            reference_scan_array(5) = Scan('data/absBlock_noFilter_July16/scans/blank_36W/', 'block36w_', 2000, 2000, 20, 80, 36, 708);
            reference_scan_array(6) = Scan('data/absBlock_noFilter_July16/scans/blank_44W/', 'block44w_', 2000, 2000, 20, 80, 44, 708);

            reference_scan_array(2).addARTistFile('data/absBlock_noFilter_July16/sim/blank/a10.tif');
            reference_scan_array(3).addARTistFile('data/absBlock_noFilter_July16/sim/blank/a18.tif');
            reference_scan_array(4).addARTistFile('data/absBlock_noFilter_July16/sim/blank/a28.tif');
            reference_scan_array(5).addARTistFile('data/absBlock_noFilter_July16/sim/blank/a36.tif');
            reference_scan_array(6).addARTistFile('data/absBlock_noFilter_July16/sim/blank/a44.tif');
            
            reference_white = 5;
        end
        
        %GET REFERENCE SCAN ARRAY Sep 16
        function [reference_scan_array,reference_white] = getReferenceScanArray_Sep16()
            reference_scan_array(5) = Scan();
            reference_scan_array(1) = Scan('data/absBlock_CuFilter_Sep16/scans/blank_0W/', '0w_', 2000, 2000, 20, 0, 0, 500);
            reference_scan_array(2) = Scan('data/absBlock_CuFilter_Sep16/scans/blank_5W/', '5w_', 2000, 2000, 20, 80, 4.96, 500);
            reference_scan_array(3) = Scan('data/absBlock_CuFilter_Sep16/scans/blank_10W/', '10w_', 2000, 2000, 20, 80, 10, 500);
            reference_scan_array(4) = Scan('data/absBlock_CuFilter_Sep16/scans/blank_15W/', '15w_', 2000, 2000, 20, 80, 15.04, 500);
            reference_scan_array(5) = Scan('data/absBlock_CuFilter_Sep16/scans/blank_20W/', '20w_', 2000, 2000, 20, 80, 20, 500);

            reference_scan_array(2).addARTistFile('data/absBlock_CuFilter_Sep16/sim/blank/a5.tif');
            reference_scan_array(3).addARTistFile('data/absBlock_CuFilter_Sep16/sim/blank/a10.tif');
            reference_scan_array(4).addARTistFile('data/absBlock_CuFilter_Sep16/sim/blank/a15.tif');
            reference_scan_array(5).addARTistFile('data/absBlock_CuFilter_Sep16/sim/blank/a20.tif');
            
            reference_white = 5;
        end
        
        %GET REFERENCE SCAN ARRAY Dec 16
        function [reference_scan_array,reference_white] = getReferenceScanArray_Dec16()
            reference_scan_array(4) = Scan();
            reference_scan_array(1) = Scan('data/titaniumBlock_SnFilter_Dec16/scans/blank_0W/', '0w_', 2000, 2000, 20, 0, 0, 1415);
            reference_scan_array(2) = Scan('data/titaniumBlock_SnFilter_Dec16/scans/blank_10W/', '10w_', 2000, 2000, 20, 190, 10.07, 1415);
            reference_scan_array(3) = Scan('data/titaniumBlock_SnFilter_Dec16/scans/blank_20W/', '20w_', 2000, 2000, 20, 190, 19.95, 1415);
            reference_scan_array(4) = Scan('data/titaniumBlock_SnFilter_Dec16/scans/blank_26W/', '26w_', 2000, 2000, 20, 190, 26.03, 1415);

            reference_scan_array(2).addARTistFile('data/titaniumBlock_SnFilter_Dec16/sim/blank/10w.tif');
            reference_scan_array(3).addARTistFile('data/titaniumBlock_SnFilter_Dec16/sim/blank/20w.tif');
            reference_scan_array(4).addARTistFile('data/titaniumBlock_SnFilter_Dec16/sim/blank/26w.tif');
            
            reference_white = 3;
        end
        
    end
    
end

