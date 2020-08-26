%MIT License
%Copyright (c) 2020 Sherman Lo

%CLASS: EMPIRICAL NULL FILTER GPU
%GPU version of the empirical null filter, makes use of ImageJ and JCuda
%HOW TO USE:
  %see superclass
classdef EmpiricalNullFilterGpu < EmpiricalNullFilter
  
  methods (Access = public)
    
    %CONSTRUCTOR
    %PARAMETERS:
      %radius: the radius of the kernel
    function this = EmpiricalNullFilterGpu(radius)
      this = this@EmpiricalNullFilter(radius);
      this.javaFilter = uk.ac.warwick.sip.empiricalnullfilter.EmpiricalNullFilterGpu();
      this.javaFilter.setRadius(radius);
    end
    
    %METHOD: SET BLOCK DIM X
    %Set the block dimension x for the GPU
    function setBlockDimX(this, blockDimX)
      this.javaFilter.setBlockDimX(blockDimX);
    end
    
    %METHOD: SET BLOCK DIM Y
    %Set the block dimension y for the GPU
    function setBlockDimY(this, blockDimY)
      this.javaFilter.setBlockDimY(blockDimY);
    end
    
  end
  
end

