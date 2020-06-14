function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

sum = 0;

for i = 1:m
  z = theta' * X(i,:)';
  yi = y(i);
  
  sum += -1*yi*log(sigmoid(z)) - (1-yi)*log(1 - sigmoid(z));
endfor

J = (1 / m) * sum;

grad = zeros(size(theta));

for i=1:length(theta)
  sumG = 0;
  for  j = 1:m
    z = theta' * X(j,:)';
    
    sumG += (sigmoid(z) - y(j)) * X(j,i);
  endfor
  
  grad(i) = (1 / m) * sumG;
endfor

% =============================================================

end
