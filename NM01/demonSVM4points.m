clc; close all; warning off

a        = 10;
A0       = [0 0; 0 1; 1 0; 1 a]; 
c        = [-1 -1  1  1]';
 
[m,n]    = size(A0);  
pars.lam = 10;
out      = NM01svm(A0,c,pars); 
x        = out.x;

figure('Renderer', 'painters', 'Position', [1000, 300,350 330])
axes('Position', [0.08 0.08 0.88 0.88] );
scatter([1;1],[0 a],80,'+','m'), hold on
scatter([0;0],[0,1],80,'x','b'), hold on
line([-x(3)/x(1) -x(3)/x(1)],[-1 1.1*a],'Color', 'r')
axis([-.1 1.1 -1 1.1*a]),box on
ld = strcat('NM01:',num2str(accuracy(A0,x,c)*100,'%.0f%%'));
legend('Positive','Negative',ld,'location','NorthWest')