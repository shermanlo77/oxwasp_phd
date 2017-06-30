function cpQQDensity(n, lambda, alpha, beta)

    X = CompoundPoisson.simulate(n,lambda,alpha,beta);
    compound_poisson = CompoundPoisson();
    compound_poisson.setParameters(lambda,alpha,beta);
    
    zero_index = (X==0);
    n_0 = sum(zero_index);
    X_no_0 = X(~zero_index);

    x_array = linspace(min(X),max(X),500);
    if x_array(1)==0
        x_array(1) = [];
    end
    pdf_range = compound_poisson.getPdf(x_array);
    p_0 = compound_poisson.getPdf(0);

    %REAL DENSITY
    
    figure;
    yyaxis left;
    h = histogram(X_no_0,'Normalization','CountDensity');
    hold on;
    
    plot(x_array,pdf_range*(n),'r-');
    ylabel('frequency density');
    
    yyaxis right;
    scatter(h.BinWidth/2,n_0,50,'filled','b');
    scatter(h.BinWidth/4,n*p_0,50,'filled','r');
    plot([h.BinWidth/2,h.BinWidth/2],[0,n_0],'b');
    plot([h.BinWidth/4,h.BinWidth/4],[0,n*p_0],'r');
    ylabel('frequency');

    xlim([min(X),max(X)]);
    xlabel('support');
    legend('Simulation freq. density','Theoretical freq. density','Simulation freq.','Theoretical freq');
    
    %SADDLE POINT
    pdf_range = compound_poisson.getSaddlePdf(x_array);
    figure;
    histogram(X,'Normalization','CountDensity');
    hold on;
    plot(x_array,pdf_range*(n),'r-');
    ylabel('frequency density');
    xlim([min(X),max(X)]);
    xlabel('support');
    legend('Simulation freq. density','Theoretical freq. density');
    
    %QQPLOT
    p = ((1:n)'-0.5)/n;
    x_theoretical = compound_poisson.getinv(p, min(X)*0.5, max(X)*1.5, 10000, false);
    figure;
    scatter(x_theoretical,sort(X),'x');
    hold on;
    plot(x_theoretical,x_theoretical);
    
    x_theoretical = compound_poisson.getinv(p, max([1E-4,min(X(zero_index))])*0.5, max(X)*1.5, 10000, true);
    figure;
    scatter(x_theoretical,sort(X),'x');
    hold on;
    plot(x_theoretical,x_theoretical);
    
end

