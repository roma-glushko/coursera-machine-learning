function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

dimention = size(z, 2);

if (dimention >= 1)
  g = 1 ./ (1 + exp(-1 * z));
else
  g = 1 ./ (1 + expm(-1 .* z));
endif

% =============================================================

end
