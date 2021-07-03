clc, close all, warning off 
 
n     = 500;            % Signal dimension 
m     = ceil(0.5*n);    % Number of measurements
s     = ceil(0.01*n);   % signal sparsity
r     = 0.05;           % probability of sign flips
v     = 0.5;
test  = 'r';
type  = 'Cor'; 
S     = 100;
switch test
    case 's',   test0 = ceil(2:10);
    case 'm/n', test0 = 0.2:0.1:1;
    case 'r',   test0 = 0.01:0.01:0.1;
    case 'v',   test0 = 0.1:0.1:0.9;
    case 'n',   test0 = 2000:2000:10000; S=10;
end 
recd   = zeros(nnz(test0),4);
for j  = 1:nnz(test0)
    switch test
        case 's',     s=test0(j);
        case 'm/n',   m=ceil(test0(j)*n);
        case 'r',     r=test0(j);
        case 'v',     v=test0(j);    
        case 'n',     n=test0(j); m=ceil(n/2); s=ceil(0.005*n); 
    end
    for ii = 1 : S    
        % --------- Generate data ---------------------
       [A0,c,co,xo] = random1bcs('Cor',m,n,s,0.1,r,v);
        pars.sp     = s;
        out         = NM01bcs(A0,c,pars);
        recd(j,1)   = recd(j,1) - 20*log10(norm(xo-out.x));    
        recd(j,2)   = recd(j,2) + nnz(sign(A0*out.x)-c)/m;
        recd(j,4)   = recd(j,4) + nnz(sign(A0*out.x)-co)/m;
        recd(j,3)   = recd(j,3) + out.time;
    end
end
 
 
recd = recd/S; 
ylab = {'SNR','HD','TIME','HE'};
figure('Renderer', 'painters', 'Position', [900, 200, 450 350])
xloc = [-0.05  0.01 -0.05  0.01];
yloc = [ 0.01  0.01 -0.02 -0.02];
for j    = 1:4
    sub  = subplot(2,2,j); 
    pos  = get(sub, 'Position'); 
    tmp  = recd(:,j);
    plot(test0,tmp,'black.-','LineWidth',0.75), hold on,
    grid on, xlabel(test), ylabel(ylab{j})  
    axis([min(test0) max(test0) min(tmp)/1.1 max(tmp)*1.1]) 
    set(sub, 'Position',pos+[xloc(j),yloc(j),0.06,0.05] )
end   
 
