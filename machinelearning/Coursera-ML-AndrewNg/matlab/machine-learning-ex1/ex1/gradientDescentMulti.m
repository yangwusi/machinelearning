function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha
%   以learning rate alpha 作为学习率使用num_iters 次梯度下降
%   Initialize some useful values 
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
n = size(X , 2);% n=3,这里表示传入数据X的列数，即是特征

%例如：x=[1,2,3;4,5,6]是一个2*3的矩阵，则：
%d = size(X);    %返回矩阵的行数和列数，保存在d中
%[m,n] = size(X)%返回矩阵的行数和列数，分别保存在m和n中
%m = size(X,dim);%返回矩阵的行数或列数，dim=1返回行数，dim=2返回列数
% 设置J值用来保存(h_theta(X^(i))-y(i))

%J=zeros(n, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %在调试时，将cost function的值和梯度输出会十分的有效
%    以下是我自己写的代码，但是验证时没有给出通过，此时将利用答案给出的代码
   
%    J = X*theta-y;
%    % 注意对于theta值的更新需要同时进行更新
%    temp1 = theta(1)-alpha*(1/m)*sum(J.*X(:,1));
%    temp2 = theta(2)-alpha*(1/m)*sum(J.*X(:,2));
%    temp3 = theta(3)-alpha*(1/m)*sum(J.*X(:,3));
%    theta(1) = temp1;
%    theta(2) = temp2;
%    theta(3) = temp3;
T = zeros(n,1); 
H = X * theta;
for i = 1 : m,
	T = T + (H(i) - y(i)) * X(i,:)';	
end
	
theta = theta - (alpha * T) / m;









    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
