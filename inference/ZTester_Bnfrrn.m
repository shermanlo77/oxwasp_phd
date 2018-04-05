%CLASS: Z TESTER bonferroni
classdef ZTester_Bnfrrn < ZTester
    
    properties (SetAccess = protected)
    end
    
    methods (Access = public)
        
        %CONSTRUCTOR
        function this = ZTester_Bnfrrn(z_image)
           this@ZTester(z_image); 
        end
        
        %OVERRIDE: DO TEST
        %Does hypothesis using the p values, corrected using bonferroni
        %Saves significant pixels in the member variable sig_image
        function doTest(this)
            %calculate the p values
            this.getPValues();
            %save the results, sig_image and size_corrected
            this.sig_image = this.p_image < (this.size/this.n_test); %2d boolean of significant pixels
            this.size_corrected = this.size/this.n_test; %size of test, corrected using bonferroni
        end
        
    end
    
end

