%FUNCTION: COMPOUND POISSON QQ PLOT FOR DENSITIES
%Plots histograms of simulations and pdf (exact and saddle point approximation)
%The simulations and pdf are also compared via QQ plot
%PARAMETERS:
    %n: number of simulations
    %lamdba: poisson parameter
    %alpha: gamma shape parameter
    %beta: gamma rate parameter
%NOTES:
    %for the qq plot, the quantiles are calculated by numerically integrating the pdf to get the cdf
    %then invered using interpolation
    %for the saddle point approximation, zero is not supported, integrations is done from no smaller than 1E-4
function cpQQDensity(n, lambda, alpha, beta)

    %% SET UP

    %simulated n compound poisson varibales
    X = CompoundPoisson.simulate(n,lambda,alpha,beta);
    
    %set up a compound poisson random variable
    compound_poisson = CompoundPoisson();
    compound_poisson.setParameters(lambda,alpha,beta);
    
    %get the index of data points with the value 0
    zero_index = (X==0);
    %get the number of zeros
    n_0 = sum(zero_index);
    %get non-zero data
    X_no_0 = X(~zero_index);

    %get an array of xs, equally spaced out across the range of the observed simulated data
    x_array = linspace(min(X),max(X),500);
    %if the minimum of the data is 0, remove it
    if x_array(1)==0
        x_array(1) = [];
    end
    
    %get the pdf and mass at 0
    pdf_range = compound_poisson.getPdf(x_array);
    p_0 = compound_poisson.getPdf(0);
    
    %get the saddle point approximated pdf
    pdf_saddle_range = compound_poisson.getSaddlePdf(x_array);

    %% REAL DENSITY
    
    figure;
    
    %plot the histogram (frequency density axis on the left)
    subplot(2,2,1);
    yyaxis left;
    h = histogram(X_no_0,'Normalization','CountDensity');
    hold on;
    %plot the pdf
    plot(x_array,pdf_range*(n),'r-');
    ylabel('frequency density');
    
    %plot the zero frequency for both the simulation and the pmf (frequency axis on the right)
    yyaxis right;
    %plot the frequency in the first bin of the histogram
    scatter(h.BinWidth/2,n_0,50,'filled','b'); %plot the simulation as a circle
    scatter(h.BinWidth/4,n*p_0,50,'filled','r'); %plot the pmf as a circle
    plot([h.BinWidth/2,h.BinWidth/2],[0,n_0],'b'); %simulation line
    plot([h.BinWidth/4,h.BinWidth/4],[0,n*p_0],'r'); %pmf line
    ylabel('frequency');

    %set the size of the x axis and label
    xlim([min(X),max(X)]);
    xlabel('support');
    legend('Simulation freq. density','Theoretical freq. density','Simulation freq.','Theoretical freq');
    
    %% SADDLE POINT
    
    %plot the histogram
    subplot(2,2,2);
    histogram(X,'Normalization','CountDensity');
    hold on;
    %plot the saddle point approximation pdf
    plot(x_array,pdf_saddle_range*(n),'r-');
    %set the size of the x axis and label
    ylabel('frequency density');
    xlim([min(X),max(X)]);
    xlabel('support');
    legend('Simulation freq. density','Theoretical freq. density');
    
    %% QQPLOT
    
    %get array of percentages
    p = ((1:n)'-0.5)/n;
    
    %for the exact pdf, get the quantiles
    %start the numerical integration from 0.5*min(X) to 1.5*max(X) using 10000 trapeziums
    x_theoretical = compound_poisson.getinv(p, min(X)*0.5, max(X)*1.5, 10000, false);
    subplot(2,2,3);
    %plot the simulated quantiles vs the exact quantiles
    scatter(x_theoretical,sort(X),'x');
    hold on;
    %plot straight line
    plot(x_theoretical,x_theoretical);
    xlabel('Theoretical quantiles');
    ylabel('Simulation quantiles');
    xlim([x_theoretical(1),x_theoretical(end)]);
    
    %for the saddle point pdf, get the quantiles
    %start the numerical integration from max([1E-4,min(X(zero_index))*0.5]) to 1.5*max(X) using 10000 trapeziums
    x_theoretical = compound_poisson.getinv(p, max([1E-4,min(X(zero_index))*0.5]), max(X)*1.5, 10000, true);
    subplot(2,2,4);
    %plot the simulated quantiles vs the saddle quantiles
    scatter(x_theoretical,sort(X),'x');
    hold on;
    %plot straight line
    plot(x_theoretical,x_theoretical);
    xlabel('Theoretical quantiles');
    ylabel('Simulation quantiles');
    xlim([x_theoretical(1),x_theoretical(end)]);
    
end

