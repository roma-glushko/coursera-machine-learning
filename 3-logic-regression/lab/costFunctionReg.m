function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

sum = 0;

for i = 1:m
  z = theta' * X(i,:)';
  yi = y(i);
  
  sum += -1*yi*log(sigmoid(z)) - (1-yi)*log(1 - sigmoid(z));
endfor

regSum = 0;

for i=2:n
  regSum += theta(i) ^ 2;
endfor

J = (1 / m) * sum + (lambda/(2*m)) * regSum;

grad = zeros(size(theta));

for i=1:n
  sumG = 0;
  for  j = 1:m
    z = theta' * X(j,:)';
    
    sumG += (sigmoid(z) - y(j)) * X(j,i);
  endfor
  
  grad(i) = (1 / m) * sumG;

  if (i>1)
    grad(i) += (lambda / m) *  theta(i);
  endif
endfor

% =============================================================

end
