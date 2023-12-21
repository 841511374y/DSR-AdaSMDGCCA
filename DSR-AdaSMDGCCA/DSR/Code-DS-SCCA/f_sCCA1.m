function [w1, w2, obj] = f_sCCA(X, Y, paras)




% set parameters
t1 = paras.lambda1;
t2 = paras.lambda2;

% initialize canonical loadings
n_XVar = size(X,2);
n_YVar = size(Y,2);
w1 = ones(n_XVar, 1)./n_XVar;
w2 = ones(n_YVar, 1)./n_YVar;

% stop criteria
stop_err = 10e-4;


max_iter = 100;
for iter = 1:max_iter
    
    % fix w2, get w1
    res = Y*w2;   
    XX = X'*X;
    XY = X'*res;    
    Wi = sqrt(sum(w1.*w1,2)+eps);
    D1 = diag(1./Wi);
    w1 = (XX+t1*D1)\XY;
    scale1 = sqrt((X*w1)' * X * w1);
    w1 = w1 / scale1;
    
    % fix w1, get w2
    res = X*w1;
    XX = Y' * Y;
    XY = Y' * res;
    Wi = sqrt(sum(w2.*w2,2)+eps);
    D1 = diag(1./Wi);
    w2 = (XX+t2*D1)\XY;
    scale2 = sqrt((Y*w2)' * Y * w2);
    w2 = w2 / scale2;
          
     obj(iter) = w1'* X' * Y * w2 - 0.5*norm(X*w1,2) - 0.5*norm(Y*w2,2)- t1*sum(abs(w1)) - t2*sum(abs(w2));

    
    if iter > 2 && abs(obj(iter) - obj(iter-1)) < stop_err
        break;
    end
    
end
