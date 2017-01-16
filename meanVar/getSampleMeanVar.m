function [sample_var,sample_mean] = getSampleMeanVar(folder_location,length)

    block_data = BlockData();
    stack = block_data.loadSampleStack(folder_location);
    
    lower_index = (1996/2)-length;
    upper_index = (1996/2)+length;
    
    stack = stack(lower_index:upper_index,lower_index:upper_index,:);
    sample_var = reshape(var(stack,[],3),[],1);
    sample_mean = reshape(mean(stack,3),[],1);
    
    figure;
    imagesc(mean(stack,3));
    colormap gray;

end

