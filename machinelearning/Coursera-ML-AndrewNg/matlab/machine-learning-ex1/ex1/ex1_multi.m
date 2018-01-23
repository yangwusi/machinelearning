%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear regression exercise. 
%
%  You will need to complete the following functions in this 
%  exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%

%% Initialization

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);
n = size(X, 2);
% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');
fprintf('the size of data=%.0f\n',n);

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean 缩放特征并将其设置到0为均值
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

% Add intercept term to X 向X添加截距项
X = [ones(m, 1) X];


%% ================ Part 2: Gradient Descent ================

% ====================== YOUR CODE HERE ======================
% Instructions: We have provided you with the following starter
%               code that runs gradient descent with a particular
%               learning rate (alpha). 
%               我们已经向你提供了以下的初始代码可以使用一个特定的学习率alpha完成梯度下降
%
%               Your task is to first make sure that your functions - 
%               computeCost and gradientDescent already work with 
%               this starter code and support multiple variables.
%               你的任务是首先确保你的支持多变量的函数计算cost和梯度下降不成问题
%               After that, try running gradient descent with 
%               different values of alpha and see which one gives
%               you the best result.
%               之后，试着运行不同alpha的并且看哪一个是最好的结果
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 br house.
%               最后，你会预测1650平方英尺，3个房间的房子的价格
% Hint: By using the 'hold on' command, you can plot multiple
%       graphs on the same figure. 使用"hold on"命令，你可以将多张图片画在同一个框架中
%
% Hint: At prediction, make sure you do the same feature normalization.
% 在预测中，记得使用同一种数据标准化

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.01; %学习率
num_iters = 400; %迭代次数
% 现在我们需要对alpha进行变化后输出J的变化
alpha1 = 0.03;
alpha2 = 0.06;
alpha3 = 0.12;
alpha4 = 0.24;

% Init Theta and Run Gradient Descent 初始化theta并且运行梯度下降
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters); % 调用gradientDescentMulti函数
theta1 = zeros(3, 1);
theta2 = zeros(3, 1);
theta3 = zeros(3, 1);
theta4 = zeros(3, 1);
[theta1, J_history1] = gradientDescentMulti(X, y, theta1, alpha1, num_iters); % 调用gradientDescentMulti函数
[theta2, J_history2] = gradientDescentMulti(X, y, theta2, alpha2, num_iters); % 调用gradientDescentMulti函数
[theta3, J_history3] = gradientDescentMulti(X, y, theta3, alpha3, num_iters); % 调用gradientDescentMulti函数
[theta4, J_history4] = gradientDescentMulti(X, y, theta4, alpha4, num_iters); % 调用gradientDescentMulti函数
% Plot the convergence graph 画出收敛图像
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2); % numel(J_history)表示数组长度
hold on;
plot(1:numel(J_history1), J_history1, '-r', 'LineWidth', 2); % numel(J_history1)表示数组长度
plot(1:numel(J_history2), J_history2, '-k', 'LineWidth', 2); % numel(J_history2)表示数组长度
plot(1:numel(J_history3), J_history3, '-g', 'LineWidth', 2); % numel(J_history3)表示数组长度
plot(1:numel(J_history4), J_history4, '-y', 'LineWidth', 2); % numel(J_history4)表示数组长度
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
price = 0; % You should change this
x1=[1,1650,3];
x1 =[X;x1];
[x1 mu1 sigma1]= featureNormalize(x1);
length_x1=length(x1);
x2=x1(length_x1,:); % 这里我们的x2数据出现了nan的问题应该是得到的数值太小成为了0，所以我们此时应该将nan转换为0
x2(isnan(x2))=0;
price = x2*theta;

% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n');

% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form 
%               solution for linear regression using the normal
%               equations. You should complete the code in 
%               normalEqn.m
%               下面是用正规方程的近似方法进行就算而不用梯度下降的方式
%               After doing so, you should complete this code 
%               to predict the price of a 1650 sq-ft, 3 br house.
%

%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
price = 0; % You should change this
x1=[1,1650,3];
price = x1*theta;
% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);

