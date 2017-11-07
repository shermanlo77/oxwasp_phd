%CLASS: K FOLDER
%Class for returning the index of a training set and test set
%It can also rotate the sets about in a k-fold fashion
classdef KFolder < handle
    
    %MEMBER VARIABLES
    properties (SetAccess = protected)
        index; %permutation of integers from 1 to n
        k; %number of folds
        n; %size of data
        n_fold; %size of a fold (not an integer, needs to be rounded)
        i_rotate; %integer pointer, 1,2,...,k, points to which fold this is currently at
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        %PARAMETERS:
            %k: number of folds
            %n: size of data
            %n_train: size of training set
        function this = KFolder(k,n)
            %assign member variables
            this.index = randperm(n);
            this.k = k;
            this.n = n;
            this.n_fold = n/k;
            this.i_rotate = 1;
        end
        
        %METHOD: GET TRAINING SET
        %Return the index of the training set
        function train_index = getTrainingSet(this)
            train_index = this.index;
            train_index( round((this.i_rotate-1)*this.n_fold + 1) :  round(this.i_rotate*this.n_fold) ) = [];
        end
        
        %METHOD: GET TEST SET
        %Return the index of the test set
        function test_index = getTestSet(this)
            test_index = this.index( round((this.i_rotate-1)*this.n_fold + 1) :  round(this.i_rotate*this.n_fold) );
        end
        
        %METHOD ROTATE FOLDS
        %Rotate the training sets and test set around in a k-fold fashion
        function rotateFolds(this)
            %if this.i_rotate is at the end, set it to one
            if this.i_rotate == this.k
                this.i_rotate = 1;
            %else increment this.i_rotate
            else
                this.i_rotate = this.i_rotate + 1;
            end
        end
        
    end
    
end

