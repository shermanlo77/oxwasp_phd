function [P,f_array] = fft_leastSquares(x, sampling_period)

    if isrow(x)
        x = x';
    end

    if nargin == 1
        sampling_period = 1;
    end

    n = numel(x);
    L = n * sampling_period;
    
    f_array = (1:floor(n/2))/L;
    
    t = ((1:n)*sampling_period)';
    
    n_freq = numel(f_array);
    
    T = zeros(n,1+2*n_freq);
    T(:,1) = 1;
    
    for i = 1:n_freq
        
        f = f_array(i);
        T(:,1+(i-1)*2+1) = cos(2*pi*f*t);
        T(:,1+(i-1)*2+2) = sin(2*pi*f*t);
        
    end
    b = T\x;
    
    P = zeros(1,n_freq+1);
    for i = 1:n_freq
        P(i+1) = b(1+(i-1)*2+1)^2*b(1+(i-1)*2+2)^2;
    end
    P = sqrt(P);
    P(1) = b(1);

end

