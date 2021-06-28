function RecoverShow(xo,x,ind)

    figure('Renderer', 'painters', 'Position', [1000, 500, 400 200])
    axes('Position', [0.06 0.12 0.89 0.75] );
    stem(find(xo),xo(xo~=0),'bo-','MarkerSize',7, 'LineWidth',1),hold on
    stem(find(x),x(x~=0),'r*:', 'MarkerSize',5, 'LineWidth',1),hold on
    grid on, ymin= -0.1; ymax=0.2;
    xx  = [xo; x];
    if nnz(xx<0)>0, ymin= min(xx(xx<0))-0.1; end
    if nnz(xx>0)>0, ymax= max(xx(xx>0))+0.1; end   
    axis([1 length(x)  ymin ymax])
    if ind
       snr   = -20*log10(norm(x-xo)); 
       title(strcat('SNR=',num2str(snr,4)))
       set(0,'DefaultAxesTitleFontWeight','normal');
       legend('Ground-Truth', 'Recovered')
    end
end

