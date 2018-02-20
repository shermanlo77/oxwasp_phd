%MEAN VARIANCE ESTIMATOR
%Estimates the within pixel mean and variance greyvalue, for each pixel
%
%The segmentation is passed to the constuctor
%The (shading corrected) greyvalue of each masked pixel and each image is stored by calling saveGreyvalueArray
%The within pixel mean and variance greyvalue are estimated for a pixel via the method getMeanVar
%the method getMeanVar requires which images to be used in the mean and variance estimation
classdef MeanVarianceEstimator < handle

    %MEMBER VARIABLES
    properties (SetAccess = private)
        %array of masked greyvalues for each image
            %dim 1: for each pixel
            %dim 2: for each image
        greyvalue_array;
        
        %segmentation vector (column vector of length n_pixel)
        %true for ROI
        segmentation;
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        %PARAMETERS:
            %scan: scan object containing images
        %Saves the segmentation
        %NOTES:
            %shading correction should be applied prior to construction
        function this = MeanVarianceEstimator(scan)
            %get the segmentation of the scan
            %reshape the segmentation to be a column vector
            this.segmentation = reshape(scan.getSegmentation(),[],1);
            %load the images and reshape it to be a design matrix
            image_stack = scan.loadImageStack();
            image_stack = reshape(image_stack,scan.area,scan.n_sample);
            %segment the design matrix and save it to greyvalue_array
            this.greyvalue_array = image_stack(this.segmentation,:);
        end
        
        %GET MEAN VARIANCE
        %Get mean and variance vector using the images indicated by the parameter data_index
        %The mean and variance are already segmented
        %PARAMETERS:
            %image_index: vector of integers, points to which images to use for mean and variance estimation
        %RETURNS:
            %sample_mean: mean vector
            %sample_var: variance vector
        function [sample_mean,sample_var] = getMeanVar(this, image_index)
            %work out the mean and variance
            sample_mean = mean(this.greyvalue_array(:,image_index),2);
            sample_var = var(this.greyvalue_array(:,image_index),[],2);
        end
        
    end
    
end

