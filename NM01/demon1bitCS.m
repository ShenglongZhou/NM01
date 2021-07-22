clc; close all; clear all; warning off

n          = 2000; 
m          = ceil(0.5*n);
s          = ceil(0.01*m);   % sparsity level
r          = 0.01;           % flipping ratio
v          = 0.5;
[A,c,co,xo]= random1bcs('Cor',m,n,s,0.1,r,v);

pars.sp    = s; 
pars.q     = 0.5;
out        = NM01bcs(A,c,pars);

RecoverShow(xo,out.x,1)
fprintf('Computational time:    %.3fsec\n',out.time);
fprintf('Signal-to-noise ratio: %.2f\n',-20*log10(norm(out.x-xo)));
fprintf('Hamming distance:      %.3f\n',nnz(sign(A*out.x)-c)/m)
fprintf('Hamming error:         %.3f\n',nnz(sign(A*out.x)-co)/m)
