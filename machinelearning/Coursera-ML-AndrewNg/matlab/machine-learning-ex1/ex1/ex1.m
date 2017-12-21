% Machine Learning Online Class - Exercise 1: Linear Regression

%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions
%  in this exericse:
%
%     warmUpExercise.m 温习
%     plotData.m 显示数据
%     gradientDescent.m 梯度下降
%     computeCost.m 代价函数
%     gradientDescentMulti.m 多变量梯度下降
%     computeCostMulti.m  多变量代价函数
%     featureNormalize.m 特征标准化
%     normalEqn.m 正则方程法
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%
% x refers to the population size in 10,000s (人口数量)
% y refers to the profit in $10,000s (利润)
%

%% Initialization
clear ; close all; clc

%% ==================== Part 1: Basic Function ====================
% Complete warmUpExercise.m
fprintf('Running warmUpExercise ... \n');
fprintf('5x5 Identity Matrix: \n');
warmUpExercise()

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ======================= Part 2: Plotting =======================
fprintf('Plotting Data ...\n')
data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples

% Plot Data
% Note: You have to complete the code in plotData.m
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =================== Part 3: Cost and Gradient descent ===================

X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

% Some gradient descent settings
iterations = 1500; % 设置迭代次数为1500次
alpha = 0.01; % 设置学习率为0.01

fprintf('\nTesting the cost function ...\n')
% compute and display initial cost
J = computeCost(X, y, theta);
fprintf('With theta = [0 ; 0]\nCost computed = %f\n', J);
fprintf('Expected cost value (approx) 32.07\n');

% further testing of the cost function
J = computeCost(X, y, [-1 ; 2]);
fprintf('\nWith theta = [-1 ; 2]\nCost computed = %f\n', J);
fprintf('Expected cost value (approx) 54.24\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nRunning Gradient Descent ...\n')
% run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations);

% print theta to screen
fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', theta);
fprintf('Expected theta values (approx)\n');
fprintf(' -3.6303\n  1.1664\n\n');

% Plot the linear fit
hold on; % keep previous plot visible 原先的图像保持可视化,即点阵图
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure 不再对图像上的图片进行操作

% Predict values for population sizes of 35,000 and 70,000
% 预测当人口是35000和70000时城市的利润值
predict1 = [1, 3.5] *theta;
fprintf('For population = 35,000, we predict a profit of %f\n',...
    predict1*10000);
predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============= Part 4: Visualizing J(theta_0, theta_1) =============
fprintf('Visualizing J(theta_0, theta_1) ...\n')

% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = computeCost(X, y, t);
    end
end


% Because of the way meshgrids work in the surf command, we need to
% transpose J_vals before calling surf, or else the axes will be flipped
% 这个surf函数中的对[x,y]绑定的函数meshgrid函数有关,我们需要将J_vals在surf函数之前进行转置,
% 否则我们的维数会颠倒.
% 例如,对于surf(x, y, z=f(x,y));函数中的(x,y)来说,是由meshgrid函数绑定生成.
% 此函数[x,y]=meshgrid(1:0.5:2,1:6).
% x.shape=(6,3),y.shape=(6,3)
% [X,Y] = meshgrid(1:0.5:2,1:6);
% disp (X);
%  1.0000   1.5000   2.0000
%  1.0000   1.5000   2.0000
%  1.0000   1.5000   2.0000
%  1.0000   1.5000   2.0000
%  1.0000   1.5000   2.0000
%  1.0000   1.5000   2.0000
% disp (Y);
% 1   1   1
% 1   1   1
% 1   1   1
% 1   1   1
% 1   1   1
% 1   1   1

J_vals = J_vals';
% Surface plot(曲面图)
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot(等高线图)
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
% 绘图的椭圆是15个等高线，在0.01到100之间分成20份
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
