clc; close all; warning off

prob     = 'fabc'; %'arce'
samp     = load(strcat(prob, '.mat')); 
label    = load(strcat(prob, '_class.mat')); 
X        = samp.X;
y        = label.y;
y(y~=1)  = -1;   
[m,n]    = size(X); 
pars.lam = 10;
out      = NM01svm(X,y,pars); 
Acc      = accuracy(X,out.x,y);

fprintf('Training  Size:       %d x %d\n', m,n);
fprintf('Training  Time:       %5.3fsec\n',out.time);
fprintf('Training  Accuracy:   %5.2f%%\n', Acc*100) 
fprintf('Training  Objective:  %5.3e\n',   out.obj);


 