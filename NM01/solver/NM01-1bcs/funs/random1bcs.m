function [X,yf,y,xopt]=random1bcs(type,m,n,s,nf,r,v)
% This file aims at generating data of 2 types of Guassian samples 
% for 1-bit compressed sensing
% Inputs:
%       type    -- can be 'Ind','Cor' 
%       m       -- number of samples
%       n       -- number of features
%       s       -- sparsity of the true singnal, e.g., s=0.01n
%       nf      -- nosie factor 
%       r       -- flipping ratio, 0~1, e.g., r=0.05
%       v       -- corrolation factor, 0~1, e.g., v=0.5
% Outputs:
%       X       --  samples data, m-by-n matrix
%       xopt    --  n-by-1 vector, i.e., the ture singnal
%       y       --  m-by-1 vector, i.e., sign(X*xopt+noise)
%       yf      --  m-by-1 vector, y after flapping some signs
%
% written by Shenglong Zhou, 19/07/2020
switch type
    case 'Ind'
          X = randn(m,n);
    case 'Cor'
          S = v.^(abs((1:n)-(1:n)'));
          X = mvnrnd(zeros(n,1),S,m); 
end
 
[xopt,T] = sparse(n,s);
y        = sign(X(:,T)*xopt(T)+nf*randn(m,1));
yf       = flip(y,r);

end

% generate a sparse vector ------------------------------------------------
function [x,T] = sparse(n,s)
          I    = randperm(n);
          x    = zeros(n,1);
          T    = I(1:s);
          x(T) = randn(s,1);
          x(T) = x(T) + sign(x(T));
          x(T) = x(T)/norm(x(T));
end

% flip the signs of a vector ----------------------------------------------
function yf   = flip(yopt,r)
         yf   = yopt;
         m    = length(yopt);
         I    = randperm(m);
         T    = I(1:ceil(r*m));
         yf(T)= -yf(T);
end
