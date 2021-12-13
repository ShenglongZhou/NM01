function out = NM01svm(A0,c,pars)
% This code aims at solving the support vector machine with form
%
%      min  ||Dx||^2 + lam * sum_{i=1}^m l_{0/1}(1-c_i<a_i,x>)
%
% where D is a diagonal matrix with D_{11}=...=D_{n-1,n-1}=1, D_{nn}=w>0
% lam > 0 
% l_{0/1}(t) = 1 if t > 0 and 0 if t <= 0
% c_i is the i-th entry of c
% a_i = (a0_i;1) \in \R^n with a0_i being the i-th row of A0
% =========================================================================
% Inputs:
%     A0  : The sample data in \R^{m-by-(n-1)}   (required)
%     c   : The classes of the sample in \R^m    (required)
%           c_i \in {+1,-1}, i=1,2,...,m             
%     pars: Parameters are all OPTIONAL
%           pars.x0    --  The initial point     (default zeros(n,1))
%           pars.lam   --  The penalty parameter (default 10)
%           pars.tau   --  A useful paramter     (default 5)
%           pars.maxit --  Maximum number of iterations (default 1000)  
%           pars.tol   --  Tolerance of the halting condition (default 1e-6*sqrt(n*m)) 
% =========================================================================
% Outputs:
%     Out.x:      The solution 
%     Out.obj:    The objective function value
%     Out.time    CPU time
%     Out.iter:   Number of iterations
%     Out.acc:    Classification accuracy
% =========================================================================
% Written by Shenglong Zhou on 27/06/2021 based on the algorithm proposed in
%     Shenglong Zhou, Lili Pan, Naihua Xiu, Houduo Qi, 
%     Quadratic convergence of smoothing Newton's method for 0/1 loss optimization,
%     SIAM Journal on Optimization, 31(4): 3184–3211, 2021.
% Send your comments and suggestions to <<< slzhou2021@163.com >>>                                  
% WARNING: Accuracy may not be guaranteed!!!!!  
% =========================================================================
warning off;
t0    = tic; 
[m,n] = size(A0);
nc    = -c;
if  n < 1e4
    A = nc.*[A0 ones(m,1)];
else
    A = [spdiags(nc,0,m,m)*A0 nc];    
end
Fnorm = @(var)norm(var,'fro')^2;
n     = n+1;

[maxit,tau,w,mu,x,cgtol] = GetParameters(m,n);
if nargin<3 || ~isfield(pars,'acc');  pars.acc = 1; end
if isfield(pars,'maxit');  maxit = pars.maxit;      end
if isfield(pars,'tau');    tau   = pars.tau;        end
if isfield(pars,'x0');     x     = pars.x0;         end


z       = ones(m,1);    
ACC     = zeros(maxit,1); 
if Fnorm(x)==0
    Ax  = z;
else
    Ax  = A*x+1;    
end
Axz     = Ax+tau*z;   
H       = [ones(n-1,1); w];
H1      = [ones(n-1,1); 1/w];
maxAcc  = 0;
    
fprintf('------------------------------------------\n');
fprintf('  Iter          Accuracy          CPUTime \n')
fprintf('------------------------------------------\n');

if  isfield(pars,'lam')    
    lam  = max(1/2/tau,pars.lam);     
else
    lam  = Initialam(m,n,Axz,tau);
end
 
if issparse(A)
    sp = nnz(A)/m/n; 
    if  (sp > 0.2 && n<100) || sp >0.4
        A = full(A);
    end
end
isA = issparse(A);
 
for iter     = 1:maxit
    
    [T,empT,lam] = Ttau(Axz,Ax,tau,lam); 
    g        = x; 
    g(n)     = w*x(n); 
    
    if  empT
        raw  = 0;
    else 
        P    = A(T,:);
        zJ   = z(T);
        tmp1 = g + (zJ'*P)';
        tmp2 = Ax(T);
        raw  = Fnorm(tmp2)/m;      
    end
     
    
    sb        = sign(nc.*(Ax-1)); 
    sb(sb==0) = -1;
    acc       = 1-nnz(sb+nc)/m; 
    x0        = x;
    if iter > 1 && acc<min(0.5,ACC(iter-1))
    acc       = 1-acc; 
    x0        = -x;
    end
    
    ACC(iter)    = acc; 
    if ACC(iter)>= maxAcc
       maxAcc    = ACC(iter);
       maxx      = x0; 
       maxiter   = iter;
    end
    
    fprintf('  %3d           %8.5f          %.3fsec\n',iter,acc,toc(t0)); 
      
    stop1 = (iter> 4    && abs(acc-ACC(iter-1))<=5e-5);    
    stop2 = (raw < 1e-1 && maxiter < iter-4);
    stop3 = (raw < 1e-8 && acc>0.5);
    if stop1  || ( acc>= pars.acc-1e-5) || stop2 || stop3
       break;
    end   
     
    if  empT
        u     = - g;
        v     = - z;
    else           
        rhs    = -mu*tmp1-(tmp2'*P)';
        if  n  > 5e3 || (m<=n && n>2e3)             
            if  m > 1e3       
                fx   = @(var)( ((P*var)'*P)'+ mu*H.*var ); 
                u    = my_cg(fx,rhs,cgtol,25,zeros(n,1)); 
                v    = -z; 
                v(T) = (tmp2+P*u)/mu;  
            else
                nT = nnz(T);
                if  isA && nT>500
                    fx = @(var)( P*(H1.*(var'*P)')+ mu * var ); 
                    vT = my_cg(fx,tmp2-P*(H1.*tmp1),cgtol,25,zeros(nT,1));  
                else            
                    D   = P*(H1.*P'); 
                    D(1:nT+1:end)=D(1:nT+1:end)+mu; 
                    vT  = D\ (tmp2 - P*(H1.*tmp1)); 
                end            
                v   = -z; 
                v(T)= vT; 
                u   = - H1.*(tmp1 + (vT'*P)');  
            end
        else
            if (isA  && n>501) || n>2e3
               fx = @(var)( ((P*var)'*P)'+ mu*H.*var ); 
               u = my_cg(fx,rhs,cgtol,25,zeros(n,1));   
            else
                D = P'*P;  
                D(1:n+1:end) = D(1:n+1:end) + mu*H'; 
                u = D\rhs;   
            end
            v    = -z;
            v(T) = (tmp2+P*u)/mu; 
        end
    end 
    
    x   = x + u; 
    z   = z + v; 
    Ax  = A*x + 1;
    
    if mod(iter,5)==0   
       mu  = max(1e-10,mu/2);            
       tau = max(1e-4,tau/1.5);
       lam = lam*1.1;
    end         
    Axz = Ax +  tau*z; 

end

fprintf('------------------------------------------\n');
x        = maxx;
out.x    = x;
out.z    = z;
out.obj  = Fnorm(H.*x);
out.time = toc(t0);
out.Acc  = maxAcc;
out.iter = iter;
clear A b A0 B0 P
end

%--------------------------------------------------------------------------
function [maxit,tau,w,mu,x0,tolcg] = GetParameters(m,n)
maxit = 1e3;
w     = 1e-8;
mn    = max(m,n);
tolcg = 1e-10*sqrt(mn);
tau   = 5;
mu    = 0.001*(mn<500) + ...
        0.01*(mn>=500 & m<n) + ...
        10*(mn>=500 & m>n); 
x0    = zeros(n,1); 
end


%select the index set T----------------------------------------------------
function [T,empT,lam] = Ttau(Axz,Ax,tau,lam)

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

% Set initial lam----------------------------------------------------------
function  lam  = Initialam(m,n,z,tau)
    zp   = z(z>0);
    s    = min([m,20*n,nnz(zp)]);  
    lam  = max(5,max((zp(s))^2,1)/2/tau);    
end


% conjugate gradient-------------------------------------------------------
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
