%CLASS: Z TESTER UNCORRECTED
classdef ZTester_Uncorrected < ZTester
    
    properties (SetAccess = protected)
    end
    
    methods (Access = public)
        
        %CONSTRUCTOR
        function this = ZTester_Uncorrected(z_image)
           this@ZTester(z_image); 
        end
        
        %OVERRIDE: DO TEST
        %Does hypothesis using the p values, uncorrected for multiple hypothesis testing
        %Saves significant pixels in the member variable sig_image
        function doTest(this)
            %calculate the p values
            this.getPValues();
            %save the results, sig_image and size_corrected
            this.sig_image = this.p_image < this.size; %2d boolean of significant pixels
            this.size_corrected = this.size; %size of test
        end
        
    end
    
end

