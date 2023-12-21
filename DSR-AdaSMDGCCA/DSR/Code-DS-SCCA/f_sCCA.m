function [w1, w2, w3, obj] = f_sCCA(X, Y, Z, paras)




% set parameters
t1 = paras.lambda1;
t2 = paras.lambda2;
t3 = paras.lambda3;

% initialize canonical loadings
n_XVar = size(X,2);
n_YVar = size(Y,2);
n_ZVar = size(Z,2);
w1 = ones(n_XVar, 1)./n_XVar;
w2 = ones(n_YVar, 1)./n_YVar;
w3 = ones(n_ZVar, 1)./n_ZVar;

% stop criteria
stop_err = 10e-4;


max_iter = 100;
for iter = 1:max_iter
    
    % fix w2,w3 get w1
    res1 = Y*w2;   
    res2 = Z*w3;  
    XX = X'*X;
    XY = X'*res1;    
    XZ = X'*res2; 
    Wi = sqrt(sum(w1.*w1,2)+eps);
    D1 = diag(1./Wi);
    w1 = (XX+t1*D1)\(XY+XZ);
    scale1 = sqrt((X*w1)' * X * w1);
    w1 = w1 / scale1;
    
    % fix w1,w3 get w2
    res1 = X*w1;
    res2 = Z*w3;
    YY = Y' * Y;
    XY = Y' * res1;
    YZ = Y' * res2;
    Wi = sqrt(sum(w2.*w2,2)+eps);
    D1 = diag(1./Wi);
    w2 = (YY+t2*D1)\(XY+YZ);
    scale2 = sqrt((Y*w2)' * Y * w2);
    w2 = w2 / scale2;
    
    % fix w1,w2 get w3
    res1 = X*w1;
    res2 = Y*w2;
    ZZ = Z' * Z;
    XZ = Z' * res1;
    YZ = Z' * res2;
    Wi = sqrt(sum(w3.*w3,2)+eps);
    D1 = diag(1./Wi);
    w3 = (ZZ+t2*D1)\(XZ+YZ);
    scale3 = sqrt((Z*w3)' * Z * w3);
    w3 = w3 / scale3;
          
    
    
    
     obj(iter) = w1'* X' * Y * w2 +  w1'* X' * Z * w3  + w2'* Y' * Z * w3 - 0.5*norm(X*w1,2) - 0.5*norm(Y*w2,2) - 0.5*norm(Z*w3,2) - t1*sum(abs(w1)) - t2*sum(abs(w2)) - t2*sum(abs(w3));

    
    if iter > 2 && abs(obj(iter) - obj(iter-1)) < stop_err
        break;
    end
    
end
