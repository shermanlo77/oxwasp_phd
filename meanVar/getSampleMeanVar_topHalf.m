function [sample_var,sample_mean] = getSampleMeanVar_topHalf(folder_location,index)

    block_data = BlockData();
    stack = block_data.loadSampleStack(folder_location);
    
    stack = stack(1:998,:,index);
    sample_var = reshape(var(stack,[],3),[],1);
    sample_mean = reshape(mean(stack,3),[],1);
    
    figure;
    imagesc(mean(stack,3));
    colormap gray;

end

