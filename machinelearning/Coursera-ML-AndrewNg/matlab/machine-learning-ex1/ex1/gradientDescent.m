function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha
%   gradientDescent函数会返回利用学习率alpha经过num_iters次迭代梯度下降后的theta值
%   Initialize some useful values
%   返回值是theta和J_history的组合
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);  % 用于保存J(theta)的历史值
J = zeros(m,1); % 用J值保存(h_{theta}(x^{i})-y^{i} )
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    J = X*theta-y; % J值每次都会更新
    temp1=theta(1)-alpha*(1/m)*sum(J.*X(:,1)); 
    % theta(1)表示theta数组的第一个元素,注意J的参数是其对应的变量x.    
    temp2=theta(2)-alpha*(1/m)*sum(J.*X(:,2));
    % theta(2)表示theta数组的第二个元素,要将这两个值用temp保存便于同时更新两个变量
    theta(1)=temp1;
    theta(2)=temp2;



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta); 
    %使用cost function,将每次迭代后使用computeCost计算cost值

end

end
