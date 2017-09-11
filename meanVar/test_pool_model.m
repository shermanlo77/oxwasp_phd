clc;
close all;
clearvars;

load('results/GLMVarMean_Mar16.mat');
i_glm = 1;

scan = this.getScan();
[sample_mean,sample_var] = this.getMeanVar(1:scan.n_sample, i_glm);

tic;
y_mean = zeros(numel(sample_mean),this.n_repeat);
for i = 1:this.n_repeat
    
    model = this.glm_array(i,i_glm);
    y_mean(:,i) = model.predict(sample_mean);
    
end
y_mean = mean(y_mean,2);
toc;

tic;
model = this.glm_array(1,i_glm);
model_mean = MeanVar_GLM(model.shape_parameter,model.polynomial_order,model.link_function);
beta_array = zeros(model_mean.n_order+1,this.n_repeat);
shift_array = zeros(model_mean.n_order,this.n_repeat);
scale_array = zeros(model_mean.n_order,this.n_repeat);
y_scale_array = zeros(1,this.n_repeat);
for i = 1:this.n_repeat
    model = this.glm_array(i,i_glm);
    beta_array(:,i) = model.parameter;
    shift_array(:,i) = model.x_shift';
    scale_array(:,i) = model.x_scale';
    y_scale_array(i) = model.y_scale;
end
model_mean.y_scale = 1./sqrt(mean(y_scale_array.^(-2)));
model_mean.parameter = mean(beta_array.*repmat(y_scale_array,model.n_order+1,1),2)/model_mean.y_scale;
beta_array(1,:) = [];
model_mean.x_scale = ((model_mean.parameter(2,:)')*model_mean.y_scale ./ mean( repmat(y_scale_array,model.n_order,1) .* beta_array ./ scale_array  ,2))';
model_mean.x_shift = model_mean.x_scale.* mean(repmat(y_scale_array,model.n_order,1).*beta_array.*shift_array./scale_array,2)';
model_mean.x_shift = model_mean.x_shift./(model_mean.parameter(2,:)') / model_mean.y_scale;
y_mean_2 = model_mean.predict(sample_mean);
toc;

d = max(abs(y_mean - y_mean_2));
disp(d);