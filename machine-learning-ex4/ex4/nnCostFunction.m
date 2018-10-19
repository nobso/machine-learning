function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% one-hot encoding to make the vector into the matrix
I = eye(num_labels);
Y = zeros(size(y, 1), num_labels);
for i=1:m
  Y(i, :)= I(y(i), :);
end

% --- feedforward and cost function ---
% activation function is same as hypothesis function g(z)
% input is equivalent to the activation function at the layer 1
% 5000 X 400
a1 = X;

% adding x0 (each element's value is always 1) to the input
% making the 5000 X 400 to 5000 X 401, so that we can do matrix multiplication with Theta1 (25 X 401)
a1 = [ ones(m ,1), a1 ];
z2 = a1 * Theta1'; % Theta transpose to adjust the dimension for multiplication a1 (5000 X 401) * Theta1 (401 X 25)

% activation function at the layer 2
% activation function is same as hypothesis function g(z)
% z2 (5000 X 25)
a2 = sigmoid( z2 );

% 5000 X 26
a2 = [ones(size(z2, 1), 1) a2];

% (5000 X 26) * (26 X 10)
z3 = a2 * Theta2';
a3 = sigmoid( z3 );

% (5000 X 10)
h = a3;

unregularizedJ = sum( 1 / m * sum( (-Y) .* log(h) - ( 1-Y ) .* log( 1-h ), 2) );
J = unregularizedJ;

% regularized cost function
% calculting the penalty
% we should not be regularizing the terms that correspond to the bias
p = sum(sum(Theta1(:, 2:end).^2, 2)) + sum(sum(Theta2(:, 2:end).^2, 2));

regularizedJ = unregularizedJ + (lambda / (2 * m) * p);
J = regularizedJ;

% -------------------------------------------------------------

% backpropagation without regularization
sigma3 = a3 .- Y;
sigma2 = ( sigma3 * Theta2 ) .* sigmoidGradient([ ones(size(z2, 1), 1) z2 ]);
sigma2 = sigma2(:, 2:end);

delta1 = sigma2' * a1;
delta2 = sigma3' * a2;

Theta1_grad = delta1 / m;
Theta2_grad = delta2 / m;

% backprop with regularization
p1 = lambda / m * ([zeros(size(Theta1, 1), 1) Theta1(:, 2:end)]);
Theta1_grad = Theta1_grad + p1;

p2 = lambda / m * ([zeros(size(Theta2, 1), 1) Theta2(:, 2:end)]);
Theta2_grad = Theta2_grad + p2;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
