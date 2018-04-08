
%% Initialization
clear; close all; clc

%% Load Training Data

data = xlsread('Heatwave_train.csv');
X = data(:, [1:6]); y = data(:, 7);

%% Load Testing Data

data2 = xlsread('Heatwave_test.csv');
X2 = data2(:, [1:6]); y2 = data2(:, 7);


%% ============ Part 2: Compute Cost and Gradient ============
%  In this part, we will implement the cost and gradient
%  for logistic regression.

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);
[m2, n2] = size(X2);
% Add intercept term to x and X_test
X = [ones(m, 1) X];
X2 = [ones(m2, 1) X2];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);



%% ============= Part 3: Optimizing using fminunc  =============
%  In this part, we will use a built-in function (fminunc) to find the
%  optimal parameters theta.

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 10); % 숫자 조정하면서 정확도 찾기 

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);


%% ============== Part 4: Predict and Accuracies ==============


% Compute accuracy on our training set
p = predict(theta, X);

% Compute accuracy on our testing set
p2 = predict(theta, X2);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('Test Accuracy: %f\n', mean(double(p2 == y2)) * 100);
confusionmat(p,y)
confusionmat(p2,y2)


