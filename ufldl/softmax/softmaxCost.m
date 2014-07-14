function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

M = theta * data;
M = bsxfun(@minus, M, max(M, [], 1));
M = exp(M);
h = bsxfun(@rdivide, M, sum(M));

cost = - (1 / numCases) * sum(sum(groundTruth .* log(h))) + \
	   lambda/2*sum(sum(theta.^2));

% Comment: I can derive the gradient from the cost function J(theta).
% Just note that the factor \sigma 1{y(i)==j}log(exp(theta_j^t *
% x^(i)) / \sigma) is a bit tricky. When y(i) != j, the item still
% contributes to the gradient via a factor whose numinator does not
% depend on theta_j but its denominator does.

thetagrad = - (1 / numCases) * (groundTruth - h) * data' + lambda * theta;;

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

