function out = NM01bcs(A0,c,pars)
% This code aims at solving the support vector machine with form
%
%      min  sum_{i=1}^n (x_i^2+eps)^{q/2} + lam * sum_{i=1}^m l_{0/1}(b-c_i<a0_i,x>)
%
% eps > 0, 1 > q > 0, lam > 0, b > 0
% l_{0/1}(t) = 1 if t > 0 and 0 if t <= 0
% c_i is the i-th entry of c
% a0_i is the i-th row of A0
% =========================================================================
% Inputs:
%     A0  : The sample data in \R^{m-by-n}       (required)
%     c   : The observed sign of (A0*x) in \R^m  (required)
%           c_i \in {+1,-1}, i=1,2,...,m             
%     pars: Parameters are all OPTIONAL
%           pars.x0    --  The initial point         (default zeros(n,1))
%           pars.eps   --  Paramter in the objective (default 0.5)
%           pars.q     --  Paramter in the objective (default 0.5)
%           pars.b     --  Paramter in the objective (default 0.05)
%           pars.lam   --  The penalty parameter     (default 1)
%           pars.tau   --  A useful parameter         (default 1)
%           pars.sp    --  Sparsity level of the ture solution (default 0.01*n)) 
%           pars.maxit --  Maximum number of iterations (default 1000)  
% =========================================================================
% Outputs:
%     Out.x:      The solution 
%     Out.obj:    The objective function value
%     Out.time    CPU time
%     Out.iter:   Number of iterations
% =========================================================================
% Written by Shenglong Zhou on 27/06/2021 based on the algorithm proposed in
%     Shenglong Zhou, Lili Pan, Naihua Xiu, Houduo Qi, 2021
%     Quadratic convergence of smoothing Newton's method for 0/1 loss optimization 
% Send your comments and suggestions to <<< slzhou2021@163.com >>>                                  
% WARNING: Accuracy may not be guaranteed!!!!!  
% =========================================================================
warning off;
t0     = tic; 
[m,n]  = size(A0);
nc     = -c;
if  n  < 1e4
    A  = nc.*A0;
else
    A  = spdiags(nc,0,m,m)*A0;    
end

[maxit,lam,tau,mu,sp,b,eps,cgtol,x] = GetParameters(m,n);
if nargin<3;               pars  = [];            end
if isfield(pars,'maxit');  maxit = pars.maxit;    end 
if isfield(pars,'lam') ;   lam   = pars.lam;      end 
if isfield(pars,'tau');    tau   = pars.tau;      end
if isfield(pars,'mu');     mu    = pars.mu;       end
if isfield(pars,'b');      b     = pars.b;        end
if isfield(pars,'eps');    eps   = pars.eps;      end
if isfield(pars,'x0');     x     = pars.x0;       end
if isfield(pars,'sp');     sp    = ceil(pars.sp); end 
  
lam         = max(lam, b^2/2/tau);  
[grad,hess] = def_func(pars);
Fnorm       = @(var)norm(var,'fro')^2;
z           = ones(m,1);    
ACC         = zeros(maxit,1); 
if  norm(x) == 0
    Ax      = b*ones(m,1);
else
    Ax      = A*x + b;    
end
Axz         = Ax;   
maxAcc      = 0; 

fprintf('------------------------------------------\n');
fprintf('  Iter          Accuracy          CPUTime \n')
fprintf('------------------------------------------\n');

for iter      = 1:maxit
     
    [T,empT]  = Ttau(Axz,Ax,tau,lam); 
    g         = grad(x,eps);
    nT        = nnz(T);  
    
    if  empT
        tmp1 = g;
        raw  = 0; 
        err  = Fnorm(g) + raw  + Fnorm(z);
    else      
        P    = A(T,:);
        zT   = z(T);
        tmp1 = g + (zT'*P)'; 
        tmp2 = Ax(T);
        raw  = Fnorm(tmp2);     
        err  = Fnorm(tmp1) + raw +  (Fnorm(z) -Fnorm(zT) ); 
    end
        
    sb        = sign(nc.*(Ax-b)); 
    sb(sb==0) = -1;
    acc       = 1-nnz(sb+nc)/m; 
    x0        = x;
    if iter   > 1 && acc<min(0.5,ACC(iter-1))
       acc    = 1-acc; 
       x0     = -x;
    end
    
    ACC(iter)    = acc; 
    if ACC(iter) > maxAcc
       maxAcc    = ACC(iter);
       maxx      = x0;  
    end
    
    stop1 = (iter>5 && acc<ACC(iter-1) && min(ACC(iter-2:iter-1))==1);    
    stop2 = (iter>5 && acc>0.99995 && raw < 1e-5*sqrt(n) && n>500); 
    
    if ~stop1
        fprintf('  %3d           %8.5f          %.3fsec\n',iter,acc,toc(t0));  
    end
    
    if  isfield(pars,'sp')
        if (err<1e-4 || stop2) || stop1 ; break;  end   
    else
        if (err<1e-4 && stop2) || stop1 ; break;  end  
    end
     
    if  empT
        u      = - g;
        v      = - z;
    else                
        H      = hess(x,eps); 
         
        if  nT  < n
            H1  = 1./H; 
            rhs = tmp2 - P*(H1.*tmp1);
            if  nT  < 2e3          
                D   = P*(H1.*P');            
                D(1:nT+1:end) = D(1:nT+1:end) + mu; 
                vT  = D\rhs;  
            else    
                fx  = @(var)(mu*var + P*(H1.*(var'*P)'));
                vT  = my_cg(fx,rhs,cgtol,30,zeros(nT,1)); 
            end        
            v   = -z; 
            v(T)= vT;  
            u   = - H1.*(tmp1 + (vT'*P)');      
        else
            rhs = -mu*tmp1-(tmp2'*P)';
            if  n   < 2e3 
                D   = P'*P;  
                D(1:n+1:end) = D(1:n+1:end) + mu*H';  
                u   = D\rhs;  
            else
                fx = @(var)(mu*(H.*var) + ((P*var)'*P)');  
                u  = my_cg(fx,rhs,cgtol,50,zeros(n,1)); 
            end
            v    = -z; 
            v(T) = (tmp2+P*u)/mu;  
        end
    end 
    
    x   = x + u; 
    z   = z + v;  
    Ax  = A*x + b; 
      
    Axz = Ax + tau*z;  
    eps = max(1e-5,eps/2);
    if mod(iter,5)==0   
       mu  = max(1e-10,mu/2);                                                                          
    end    
end

fprintf('------------------------------------------\n');
if acc < ACC(1); x = maxx; end 
out.lam  = z;
out.time = toc(t0);
out.iter = iter;

if isfield(pars,'sp') % refinement step
    K       = 6;
    [sx,Ts] = maxk(abs(x),sp+K-1);
    HD      = ones(1,K);
    X       = zeros(n,K); 
    if sx(sp)-sx(sp+1) <= 2e-4 
        tem    = Ts(sp:end);
        for i  = 1:K
            X(:,i)          = zeros(n,1);
            X(Ts(1:sp-1),i) = x(Ts(1:sp-1));
            X(tem(i),i)     = x(tem(i));
            X(:,i)          = X(:,i)/norm(X(:,i));
            HD(i)           = nnz(sign(A0*X(:,i))-c)/m; 
        end
        [~,i]    = min(HD); 
        out.x    = X(:,i); 
    else
        out.x           = zeros(n,1);  
        out.x(Ts(1:sp)) = x(Ts(1:sp))/norm(x(Ts(1:sp))); 
    end   
else
  x      = SparseApprox(x); 
  out.x  = x / norm(x);   
end
 
clear A b A0 B0 P
end

%--------------------------------------------------------------------------
function [maxit,lam,tau,mu,sp,b,eps,tolcg,x0] = GetParameters(m,n)
    maxit = 1e3;
    lam   = 1;
    tau   = 1; 
    mu    = 0.05;  
    sp    = ceil(0.01*n);
    b     = 0.05 ; 
    eps   = 0.5; 
    tolcg = 1e-10*sqrt(max(m,n));
    x0    = zeros(n,1);
end

%select the index set T----------------------------------------------------
function [T,empT] = Ttau(Axz,Ax,tau,lam)
    tl   = sqrt(tau*lam/2);
    T    = find(abs(Axz-tl)<tl); 
    empT = isempty(T);  
    if  empT
        zp   = Ax(Ax>=0);  
        T    = [];
        if nnz(zp)>0
            s   = ceil(0.01*nnz(zp));   
            tau = (zp(s))^2/2/lam;     
            tl  = sqrt(tau*lam/2);
            T   = find(abs(Ax-tl)<tl);          
        end
        empT = isempty(T); 
    end
end

% Define functions---------------------------------------------------------
function [grad,hess] = def_func(pars)
    %f(x) = (x^2+e)^(q/2)    
    q    = 0.5;     
    if isfield(pars,'q'); q = pars.q;  end 
    q1   = q/2-1;
    q2   = q/2-2;
    q3   = q-1;
    grad = @(t,e)( t.* ((t.*t+e).^q1) );             
    hess = @(t,e)( ((t.*t+e).^q2).*(q3*t.*t+e) ); 
end

% Conjugate gradient method-------------------------------------------------
function x = my_cg(fx,b,cgtol,cgit,x)
    r = b;
    e = sum(r.*r);
    t = e;
    for i = 1:cgit  
        if e < cgtol*t; break; end
        if  i == 1  
            p = r;
        else
            p = r + (e/e0)*p;
        end
        w  = fx(p); 
        a  = e/sum(p.*w);
        x  = x + a * p;
        r  = r - a * w;
        e0 = e;
        e  = sum(r.*r);
    end
    
end

% get the sparse approximation of x----------------------------------------
function sx = SparseApprox(x)
    n       = length(x);
    xo      = x;
    x       = x.*x;
    T       = find(x>1e-2/n);
    [sx,id] = sort(x(T),'descend'); 
    y       = 0;
    nx      = sum(x(T)); 
    nT      = nnz(T);
    t       = zeros(nT-1,1);
    for i   = 1:nT
        if y > 0.5*nx; break; end
        y    = y + sx(i); 
        if i < nT
        t(i) = sx(i)/sx(i+1);
        end
    end
    
    if  i  < nT
        j  = find(t==max(t)); 
        i  = j(1); 
    else
        i  = nT;
    end
    
    if i  > 1
        i = min(nT,10*i);
    else
        i = nT;
    end
    sx = zeros(n,1);
    sx(T(id(1:i))) = xo(T(id(1:i)));
end
